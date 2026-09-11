"""Immutable bindings for the V27 classification base checkpoints.

The classification checkpoints are the experimental boundary for the V27
baseline.  Later knownness, quality and calibration stages are allowed to
change only ``trajectory_heads``; this module records the exact base files
and the classifier-only state hash so that such changes can be checked
without rewriting the historical classification checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)

from .protocol import V27_L1_TASK_NAME, V27_L2_TASK_NAME


FROZEN_BASE_BINDING_SCHEMA = "RADAR-LN3-V27-frozen-classification-base-binding-v1"
DEFAULT_FROZEN_BASE_BINDING_PATH = Path(
    "src-v3/v27/v27_baseline/runs/V27-frozen-classification-bases-seed42.json"
)
DEFAULT_L1_BASE_CHECKPOINT = Path(
    "src-v3/v27/v27_baseline/runs/V27-known_gate-seed42/epochs/epoch-014.pt"
)
DEFAULT_L2_BASE_CHECKPOINT = Path(
    "src-v3/v27/v27_baseline/runs/V27-dbb_specialist-seed42/epochs/epoch-011.pt"
)
_HEAD_PREFIX = "trajectory_heads."
# The classifier checkpoint is immutable.  These are the *only* keys that a
# later, independently fitted open-set stage may replace.  Do not use a broad
# ``startswith`` exemption here: a future accidental module under
# ``trajectory_heads`` must not silently become mutable classifier state.
_MUTABLE_TRAJECTORY_HEAD_SUFFIXES = frozenset(
    {
        "trajectory_encoder.weight_ih_l0",
        "trajectory_encoder.weight_hh_l0",
        "trajectory_encoder.bias_ih_l0",
        "trajectory_encoder.bias_hh_l0",
        "knownness_projection.0.weight",
        "knownness_projection.0.bias",
        "knownness_projection.1.weight",
        "knownness_projection.1.bias",
        "knownness_head.weight",
        "knownness_head.bias",
        "quality_head.0.weight",
        "quality_head.0.bias",
        "quality_head.2.weight",
        "quality_head.2.bias",
    }
)
_CLASSIFIER_CONTRACT_KEYS = (
    "class_path",
    "task",
    "class_names",
    "tr_value_dim",
    "tr_missing_dim",
    "rd_quality_dim",
    "rd_quality_names",
    "course_dim",
    "rd_physical_dim",
    "micro_doppler_dim",
    "classifier_readout",
    "liquid_v2_contract",
)


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensors deterministically, independent of torch serialization."""

    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27 base state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    return canonical_sha256(parts)


def mutable_trajectory_head_keys(state: Mapping[str, Any]) -> frozenset[str]:
    """Return the exact permitted mutable state keys, failing on drift."""

    declared = {
        name for name in state if isinstance(name, str) and name.startswith(_HEAD_PREFIX)
    }
    expected = {
        f"{_HEAD_PREFIX}{suffix}" for suffix in _MUTABLE_TRAJECTORY_HEAD_SUFFIXES
    }
    if declared != expected:
        missing = sorted(expected - declared)
        unexpected = sorted(declared - expected)
        raise ValueError(
            "V27 trajectory-head state boundary differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frozenset(expected)


def classifier_state_hash(state: Mapping[str, Any]) -> str:
    """Hash every parameter/buffer outside the explicitly mutable heads."""

    mutable = mutable_trajectory_head_keys(state)
    classifier_state = {name: value for name, value in state.items() if name not in mutable}
    if not classifier_state:
        raise ValueError("V27 checkpoint has no classifier/base state")
    return tensor_state_hash(classifier_state)


def classifier_base_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the immutable classifier/recurrent contract from a model contract."""

    if not isinstance(contract, Mapping):
        raise TypeError("V27 model contract must be an object")
    missing = [key for key in _CLASSIFIER_CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(f"V27 model contract lacks classifier fields: {missing}")
    return {key: contract[key] for key in _CLASSIFIER_CONTRACT_KEYS}


def _load_checkpoint(root: Path, path: str | Path, *, task: str) -> tuple[Path, dict[str, Any]]:
    target = resolve_project_path(root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V27 frozen base checkpoint is not an object: {target}")
    payload = dict(payload)
    if payload.get("task") != task:
        raise ValueError(f"V27 frozen base task differs: expected={task}, actual={payload.get('task')}")
    if payload.get("checkpoint_kind") != "epoch":
        raise ValueError("V27 frozen classification bases must be epoch checkpoints")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"V27 frozen base checkpoint has no model_state: {target}")
    declared_state_hash = payload.get("model_state_sha256")
    if declared_state_hash != tensor_state_hash(state):
        raise ValueError(f"V27 frozen base model state hash differs: {target}")
    if payload.get("unknown_is_model_class") is not False or int(payload.get("unknown_training_exposure", -1)) != 0:
        raise ValueError("V27 frozen classification base violates the Unknown contract")
    return target, payload


def _base_record(root: Path, path: str | Path, *, task: str, expected_epoch: int) -> dict[str, Any]:
    target, payload = _load_checkpoint(root, path, task=task)
    epoch = int(payload.get("epoch", -1))
    if epoch != int(expected_epoch):
        raise ValueError(f"V27 {task} frozen epoch differs: expected={expected_epoch}, actual={epoch}")
    metrics = dict(payload.get("metrics") or {})
    assessment = dict(metrics.get("checkpoint_assessment") or {})
    if assessment.get("task") != task or int(assessment.get("epoch", -1)) != epoch:
        raise ValueError(f"V27 {task} checkpoint assessment is not bound to the epoch")
    if assessment.get("feasible") is not True:
        raise ValueError(f"V27 {task} frozen checkpoint is not marked feasible")
    # The epoch checkpoint intentionally stores only compact selection metrics.
    # The complete validation record lives in the immutable run summary.  Do
    # not silently accept a missing metric here: the Clutter floor is part of
    # the frozen-base safety contract and must be read for the exact epoch.
    summary_path = target.parent.parent / "training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"V27 {task} frozen base training summary is missing: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError(f"V27 {task} frozen base training summary is not an object: {summary_path}")
    history = list(summary.get("history") or ())
    epoch_records = [
        dict(row) for row in history
        if isinstance(row, Mapping) and int(row.get("epoch", -1)) == epoch
    ]
    if len(epoch_records) != 1:
        raise ValueError(
            f"V27 {task} frozen base summary must contain exactly one epoch-{epoch:03d} record"
        )
    epoch_record = epoch_records[0]
    if dict(epoch_record.get("checkpoint_assessment") or {}) != assessment:
        raise ValueError(
            f"V27 {task} checkpoint assessment differs from its training summary"
        )
    evaluation = dict(epoch_record.get("validation_parent_metrics") or {})
    if task == V27_L1_TASK_NAME:
        per_class = {
            str(row.get("class")): dict(row)
            for row in evaluation.get("per_class") or ()
            if isinstance(row, Mapping)
        }
        clutter = per_class.get("KNOWN_CLUTTER_EVIDENCE")
        clutter_recall = float(clutter.get("recall", float("nan"))) if clutter else float("nan")
        if clutter_recall < 8.0 / 9.0:
            raise ValueError("V27 L1 frozen base does not meet the Clutter recall floor")
    else:
        clutter_recall = None
    state = payload["model_state"]
    return {
        "task": task,
        "epoch": epoch,
        "seed": int(payload.get("seed", -1)),
        "checkpoint_path": relative_project_path(root, target),
        "checkpoint_sha256": sha256_file(target),
        "model_state_sha256": str(payload["model_state_sha256"]),
        "classifier_state_sha256": classifier_state_hash(state),
        "config_path": str(payload.get("config_path", "")),
        "config_sha256": str(payload.get("config_sha256", "")),
        "config_identity_sha256": str(payload.get("config_identity_sha256", "")),
        "manifest_path": str(payload.get("manifest_path", "")),
        "manifest_sha256": str(payload.get("manifest_sha256", "")),
        "data_registry_sha256": str(payload.get("data_registry_sha256", "")),
        "parent_split_sha256": str(payload.get("parent_split_sha256", "")),
        "feature_config_sha256": str(payload.get("feature_config_sha256", "")),
        "state_equation_sha256": str(payload.get("state_equation_sha256", "")),
        "time_policy_sha256": str(payload.get("time_policy_sha256", "")),
        "model_contract_sha256": str(payload.get("model_contract_sha256", "")),
        "class_names": list(payload.get("class_names") or []),
        "class_weights": list(payload.get("class_weights") or []),
        "clutter_recall_floor_metric": clutter_recall,
        "training_summary_path": relative_project_path(root, summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "validation_parent_metrics": evaluation,
        "metrics": metrics,
    }


def make_frozen_base_binding(
    project_root: str | Path,
    *,
    l1_checkpoint: str | Path = DEFAULT_L1_BASE_CHECKPOINT,
    l2_checkpoint: str | Path = DEFAULT_L2_BASE_CHECKPOINT,
) -> dict[str, Any]:
    """Build a binding from the two explicitly approved classification epochs."""

    root = Path(project_root).resolve()
    l1 = _base_record(root, l1_checkpoint, task=V27_L1_TASK_NAME, expected_epoch=14)
    l2 = _base_record(root, l2_checkpoint, task=V27_L2_TASK_NAME, expected_epoch=11)
    if l1["seed"] != l2["seed"]:
        raise ValueError("V27 frozen L1/L2 bases must use the same seed")
    shared = (
        "config_sha256",
        "config_identity_sha256",
        "manifest_path",
        "manifest_sha256",
        "data_registry_sha256",
        "parent_split_sha256",
        "feature_config_sha256",
        "state_equation_sha256",
        "time_policy_sha256",
    )
    for field in shared:
        if l1[field] != l2[field]:
            raise ValueError(f"V27 frozen L1/L2 base binding differs at {field}")
    payload: dict[str, Any] = {
        "schema_version": FROZEN_BASE_BINDING_SCHEMA,
        "experiment_id": "V27",
        "status": "frozen",
        "seed": l1["seed"],
        "classification_only": True,
        "trajectory_heads_may_change": True,
        "classifier_state_must_remain_identical": True,
        "l1": l1,
        "l2": l2,
        "shared_binding": {field: l1[field] for field in shared},
        "open_set_training": {
            "knownness": "not_yet_fitted",
            "quality": "not_yet_fitted",
            "calibration": "not_yet_fitted",
        },
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return payload


def verify_frozen_base_binding(
    binding: Mapping[str, Any],
    *,
    project_root: str | Path,
    require_files: bool = True,
) -> dict[str, Any]:
    """Verify binding integrity and the current bytes of both base files."""

    root = Path(project_root).resolve()
    payload = dict(binding)
    declared = payload.pop("binding_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 frozen base binding hash cannot be recomputed")
    if payload.get("schema_version") != FROZEN_BASE_BINDING_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("V27 frozen base binding schema/status differs")
    if payload.get("classification_only") is not True or payload.get("classifier_state_must_remain_identical") is not True:
        raise ValueError("V27 frozen base binding safety flags differ")
    shared = payload.get("shared_binding")
    if not isinstance(shared, Mapping):
        raise ValueError("V27 frozen base binding lacks shared provenance")
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        record = payload.get(task_to_binding_key(task))
        if not isinstance(record, Mapping):
            raise ValueError(f"V27 frozen base binding lacks {task}")
        path = resolve_project_path(root, str(record.get("checkpoint_path", "")), must_exist=require_files)
        if require_files:
            if sha256_file(path) != str(record.get("checkpoint_sha256", "")):
                raise ValueError(f"V27 frozen {task} checkpoint file hash differs")
            _, checkpoint = _load_checkpoint(root, path, task=task)
            if int(checkpoint.get("epoch", -1)) != int(record.get("epoch", -2)):
                raise ValueError(f"V27 frozen {task} checkpoint epoch differs")
            if checkpoint.get("model_state_sha256") != record.get("model_state_sha256"):
                raise ValueError(f"V27 frozen {task} model state hash differs")
            if checkpoint.get("config_sha256") != record.get("config_sha256"):
                raise ValueError(f"V27 frozen {task} config hash differs")
            if checkpoint.get("manifest_sha256") != record.get("manifest_sha256"):
                raise ValueError(f"V27 frozen {task} manifest hash differs")
            state = checkpoint.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError(f"V27 frozen {task} checkpoint has no model state")
            if classifier_state_hash(state) != record.get("classifier_state_sha256"):
                raise ValueError(f"V27 frozen {task} classifier state hash differs")
        for field, value in shared.items():
            if record.get(field) != value:
                raise ValueError(f"V27 frozen {task} differs from shared binding at {field}")
        if int(record.get("epoch", -1)) != (14 if task == V27_L1_TASK_NAME else 11):
            raise ValueError(f"V27 frozen {task} epoch differs")
        if int(record.get("seed", -1)) != int(payload.get("seed", -2)):
            raise ValueError(f"V27 frozen {task} seed differs")
    result = {**payload, "binding_sha256": declared}
    return result


def task_to_binding_key(task: str) -> str:
    if task == V27_L1_TASK_NAME:
        return "l1"
    if task == V27_L2_TASK_NAME:
        return "l2"
    raise ValueError(f"unsupported V27 base task: {task}")


def write_frozen_base_binding(
    path: str | Path,
    binding: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> Path:
    """Create the binding once, accepting only byte-identical replays.

    ``project_root`` is intentionally mandatory.  Inferring it from the
    output path is ambiguous for absolute paths and was the source of a real
    boundary bug when the binding lived at a different nesting depth.
    """

    root = Path(project_root).resolve()
    target = resolve_project_path(root, path)
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    verify_frozen_base_binding(binding, project_root=root, require_files=False)
    text = json_text(dict(binding), indent=2)
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to replace a different V27 base binding: {target}")
    if not target.exists():
        atomic_write_text(target, text)
    return target


def load_frozen_base_binding(project_root: str | Path, path: str | Path = DEFAULT_FROZEN_BASE_BINDING_PATH) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=True)
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V27 frozen base binding must be a JSON object")
    return verify_frozen_base_binding(raw, project_root=root, require_files=True)


def load_frozen_base_model(
    context: Any,
    *,
    task: str,
    device: torch.device,
    binding_path: str | Path = DEFAULT_FROZEN_BASE_BINDING_PATH,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load a frozen classifier without requiring the mutable active manifest.

    The regular classification loader intentionally binds to the current
    source manifest.  Once a classification epoch is frozen, later open-set
    implementation files are expected to change that manifest.  This loader
    therefore verifies the immutable binding and the checkpoint's own
    metadata, then checks the immutable classifier contract.  The trajectory
    heads are intentionally re-initialised because their architecture and
    supervision may evolve after classification has frozen.
    """

    from .runtime import build_v27_model

    if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
        raise ValueError(f"unsupported V27 frozen base task: {task}")
    root = Path(context.project_root).resolve()
    binding = load_frozen_base_binding(root, binding_path)
    record = binding[task_to_binding_key(task)]
    target, payload = _load_checkpoint(root, record["checkpoint_path"], task=task)
    model = build_v27_model(context, task=task, device=device)
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27 frozen base checkpoint model_state is invalid")
    checkpoint_contract = payload.get("model_contract")
    if (
        not isinstance(checkpoint_contract, Mapping)
        or canonical_sha256(dict(checkpoint_contract))
        != str(record.get("model_contract_sha256", ""))
    ):
        raise ValueError("V27 frozen base model contract hash differs")
    if classifier_base_contract(checkpoint_contract) != classifier_base_contract(
        model.model_contract()
    ):
        raise ValueError("V27 frozen base classifier contract differs from the current model")

    checkpoint_mutable = mutable_trajectory_head_keys(state)
    current_state = model.state_dict()
    current_mutable = mutable_trajectory_head_keys(current_state)
    checkpoint_base = {
        name: value for name, value in state.items() if name not in checkpoint_mutable
    }
    current_base = {
        name: value
        for name, value in current_state.items()
        if name not in current_mutable
    }
    if set(checkpoint_base) != set(current_base):
        missing = sorted(set(checkpoint_base) - set(current_base))
        unexpected = sorted(set(current_base) - set(checkpoint_base))
        raise ValueError(
            "V27 frozen base/current classifier state keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, value in checkpoint_base.items():
        current = current_base[name]
        if not isinstance(value, torch.Tensor) or not isinstance(current, torch.Tensor):
            raise TypeError(f"V27 frozen base state entry is not a tensor: {name}")
        if value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError(f"V27 frozen classifier state shape/dtype differs: {name}")
    incompatible = model.load_state_dict(checkpoint_base, strict=False)
    if set(incompatible.missing_keys) != set(current_mutable) or incompatible.unexpected_keys:
        raise ValueError(
            "V27 frozen base partial load boundary differs: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if classifier_state_hash(model.state_dict()) != record["classifier_state_sha256"]:
        raise ValueError("V27 frozen base classifier state changed during loading")
    return model, dict(payload), binding


__all__ = [
    "DEFAULT_FROZEN_BASE_BINDING_PATH",
    "DEFAULT_L1_BASE_CHECKPOINT",
    "DEFAULT_L2_BASE_CHECKPOINT",
    "FROZEN_BASE_BINDING_SCHEMA",
    "classifier_base_contract",
    "classifier_state_hash",
    "load_frozen_base_binding",
    "load_frozen_base_model",
    "make_frozen_base_binding",
    "mutable_trajectory_head_keys",
    "task_to_binding_key",
    "tensor_state_hash",
    "verify_frozen_base_binding",
    "write_frozen_base_binding",
]
