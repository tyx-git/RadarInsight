"""Independent knownness/quality fitting and dynamic V27 calibration.

The classification checkpoints are frozen before this module is entered.  The
module consequently has two deliberately separate responsibilities:

* materialize the causal trajectory features emitted by a frozen L1/L2 base;
* fit only ``trajectory_heads`` and then fit a dynamic, fail-closed L3
  registry on the independent calibration role.

No function in this module opens the test split.  Unknown labels are used only
after the frozen classification boundary and never become classifier targets.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    resolve_project_path,
    sha256_file,
    write_torch_if_absent_or_identical,
)

from .base_binding import (
    load_frozen_base_model,
    mutable_trajectory_head_keys,
    tensor_state_hash,
)
from .data import (
    V27OpenSetDataset,
    build_v27_open_set_dataloader,
    model_input_kwargs,
)
from .l3 import (
    ACTION_SELECTION_METHOD,
    make_fitted_registry,
    resolve_l3_action,
    verify_l3_registry,
)
from .logging_utils import (
    TRAINING_LOGGER_PREFIX,
    close_training_logger,
    configure_training_logger,
    format_duration,
    log_event,
    log_metric_set,
    progress_interval,
)
from .model import V27LiquidV2TaskModel
from .open_set_data import (
    load_open_set_pool_artifacts,
    open_set_role_cache_paths,
)
from .protocol import (
    V27Config,
    V27_L1_TASK_NAME,
    V27_L2_TASK_NAME,
    V27_L2_CLASS_NAMES,
    v27_artifact_path,
)
from .runtime import (
    configured_device,
    environment_snapshot,
    read_v27_sealed_json,
    seed_v27,
    write_v27_sealed_json,
    write_v27_sealed_json_if_absent_or_identical,
)


REPRESENTATION_CACHE_SCHEMA = "RADAR-LN3-V27-frozen-trajectory-representation-v1"
REPRESENTATION_MANIFEST_SCHEMA = "RADAR-LN3-V27-representation-cache-manifest-v1"
HEAD_CHECKPOINT_SCHEMA = "RADAR-LN3-V27-open-set-head-checkpoint-v1"
HEAD_SUMMARY_SCHEMA = "RADAR-LN3-V27-open-set-head-training-summary-v1"
CALIBRATION_SCHEMA = "RADAR-LN3-V27-dynamic-calibration-v3"
OPEN_SET_ROLES = ("representation_fit", "representation_development", "calibration")
CANONICAL_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
LABEL_TO_CODE = {name: index for index, name in enumerate(CANONICAL_LABELS)}


def _head_stage_contract(stage: str) -> dict[str, Any]:
    """Describe one independent head-fitting boundary without ambiguity."""

    contracts = {
        "knownness": {
            "stage": "knownness",
            "joint_training": False,
            "optimized_components": [
                "trajectory_encoder",
                "knownness_projection",
                "knownness_head",
            ],
        },
        "quality": {
            "stage": "quality",
            "joint_training": False,
            "optimized_components": ["quality_head"],
        },
        "final": {
            "stage": "final",
            "joint_training": False,
            "optimized_components": [],
            "composition": "independent_knownness_and_quality_checkpoints",
        },
    }
    try:
        return dict(contracts[str(stage)])
    except KeyError as exc:
        raise ValueError(f"unsupported V27 head stage: {stage}") from exc


def _head_component_state_hash(
    state: Mapping[str, Any], *, component: str
) -> str:
    """Hash the exact state owned by one independently trained component."""

    prefixes = {
        "knownness": (
            "trajectory_encoder.",
            "knownness_projection.",
            "knownness_head.",
        ),
        "quality": ("quality_head.",),
    }
    try:
        selected = {
            str(name): value
            for name, value in state.items()
            if str(name).startswith(prefixes[component])
        }
    except KeyError as exc:
        raise ValueError(f"unsupported V27 head component: {component}") from exc
    if not selected:
        raise ValueError(f"V27 {component} state is empty")
    return _tensor_digest(selected)


def _require_task(task: str) -> str:
    value = str(task)
    if value not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
        raise ValueError(f"unsupported V27 open-set task: {task}")
    return value


def _require_role(role: str) -> str:
    value = str(role)
    if value not in OPEN_SET_ROLES:
        raise ValueError(f"unsupported V27 open-set role: {role}")
    return value


def _cache_path(context: Any, *, task: str, role: str) -> Path:
    _require_task(task)
    _require_role(role)
    return v27_artifact_path(
        context.project_root,
        context.config.artifacts.representation_cache_template,
        int(context.config.training.seeds[0]),
        task=task,
        role=role,
    )


def _score_cache_path(context: Any, *, task: str, role: str, seed: int) -> Path:
    _require_task(task)
    _require_role(role)
    return v27_artifact_path(
        context.project_root,
        context.config.artifacts.head_score_cache_template,
        int(seed),
        task=task,
        role=role,
    )


def _head_stage_path(
    context: Any,
    *,
    task: str,
    stage: str,
    seed: int,
) -> Path:
    _require_task(task)
    if stage == "knownness":
        field = "knownness_head_checkpoint_template"
    elif stage == "quality":
        field = "quality_head_checkpoint_template"
    elif stage == "final":
        field = "head_checkpoint_template"
    else:
        raise ValueError(f"unsupported V27 head stage: {stage}")
    template = getattr(context.config.artifacts, field)
    return v27_artifact_path(
        context.project_root,
        template,
        int(seed),
        task=task,
    )


def _head_stage_summary_path(
    context: Any,
    *,
    task: str,
    stage: str,
    seed: int,
) -> Path:
    _require_task(task)
    if stage == "knownness":
        field = "knownness_head_training_summary_template"
    elif stage == "quality":
        field = "quality_head_training_summary_template"
    elif stage == "final":
        field = "head_training_summary_template"
    else:
        raise ValueError(f"unsupported V27 head stage: {stage}")
    return v27_artifact_path(
        context.project_root,
        getattr(context.config.artifacts, field),
        int(seed),
        task=task,
    )


def _head_log_path(
    context: Any,
    *,
    task: str,
    component: str,
    seed: int,
) -> Path:
    """Resolve a stage-local log without mixing knownness and quality runs."""

    if component not in {"knownness", "quality"}:
        raise ValueError("V27 head log component must be knownness or quality")
    base = v27_artifact_path(
        context.project_root,
        context.config.artifacts.head_training_log_template,
        int(seed),
        task=task,
    )
    return resolve_project_path(
        context.project_root, base.parent / component / "training.log"
    )


def _archive_derived_artifact(
    context: Any,
    path: Path,
    *,
    category: str,
) -> Path:
    """Move one obsolete derived artifact to a hash-named archive safely."""

    if category not in {"head-stages", "score-caches", "reports"}:
        raise ValueError("V27 derived artifact archive category is invalid")
    target = resolve_project_path(context.project_root, path, must_exist=True)
    archive_dir = context.project_root / "artifacts" / "archive" / "v27" / category
    archive_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(target)
    archived = archive_dir / f"{target.name}.stale-before-{digest[:16]}"
    if archived.exists():
        if sha256_file(archived) != digest:
            raise FileExistsError(f"V27 artifact archive collision: {archived}")
        target.unlink()
    else:
        target.replace(archived)
    return archived


def _archive_stale_head_stage_artifacts(
    context: Any,
    *,
    task: str,
    stage: str,
    seed: int,
    representation_manifest: Mapping[str, Any],
) -> None:
    """Clear only stale stage evidence before a fresh manifest-bound fit."""

    checkpoint_path = _head_stage_path(
        context, task=task, stage=stage, seed=seed
    )
    summary_path = _head_stage_summary_path(
        context, task=task, stage=stage, seed=seed
    )
    expected_representation = representation_manifest[
        "representation_cache_manifest_sha256"
    ]
    for path, reader in (
        (checkpoint_path, lambda value: torch.load(value, map_location="cpu", weights_only=False)),
        (summary_path, read_v27_sealed_json),
    ):
        if not path.exists():
            continue
        try:
            existing = reader(path)
            current = (
                isinstance(existing, Mapping)
                and existing.get("system_manifest_sha256") == context.manifest_sha256
                and existing.get("representation_cache_manifest_sha256")
                == expected_representation
            )
        except Exception:
            current = False
        if not current:
            _archive_derived_artifact(context, path, category="head-stages")


def _write_derived_sealed_json(
    context: Any,
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Preserve a changed report while allowing a deterministic replay reuse."""

    if path.exists():
        try:
            existing = read_v27_sealed_json(path)
            existing_payload = {
                key: value for key, value in existing.items() if key != "payload_sha256"
            }
        except Exception:
            existing_payload = None
        if existing_payload == dict(payload):
            return path
        _archive_derived_artifact(context, path, category="reports")
    write_v27_sealed_json(path, dict(payload))
    return path


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash(values)


def _representation_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27 representation cache lacks tensors")
    tensor_hash = _tensor_digest(
        {str(name): value for name, value in tensors.items()}
    )
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256({"metadata": metadata, "tensor_sha256": tensor_hash})


def _score_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27 score cache lacks tensors")
    tensor_hash = _tensor_digest({
        str(name): value for name, value in tensors.items()
    })
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256", "score_cache_sha256"}
    }
    return canonical_sha256({"metadata": metadata, "tensor_sha256": tensor_hash})


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("V27 head model has no parameters") from exc


def _head_target_weights(
    payload: Mapping[str, Any], *, task: str, config: V27Config
) -> tuple[float, float]:
    """Compute fixed fit-pool weights once, independent of minibatch makeup."""

    tensors = payload["tensors"]
    labels = [str(value) for value in payload["label_names"]]
    padding = tensors["padding_mask"]
    known_parent = _knownness_targets(task, labels).unsqueeze(1)
    known_target = known_parent.expand_as(padding).to(dtype=torch.float32)
    known_mask = _eligible_prefix_mask(
        padding, minimum_events=config.open_set.knownness_min_events
    )
    quality_target = deterministic_quality_targets(
        task,
        task_facts_sequence=tensors["task_facts_sequence"],
        event_facts_sequence=tensors["event_facts_sequence"],
        padding_mask=padding,
        config=config,
    )
    quality_mask = ~padding
    return (
        _positive_weight(known_target, known_mask),
        _positive_weight(quality_target, quality_mask),
    )


def _final_quality_targets(
    task: str, payload: Mapping[str, Any], config: V27Config
) -> np.ndarray:
    """Return final maturity labels through the same causal target helper."""

    tensors = payload["tensors"]
    lengths = tensors["lengths"]
    if lengths.ndim != 1:
        raise ValueError("V27 cached lengths must be one-dimensional")
    targets = deterministic_quality_targets(
        task,
        task_facts_sequence=tensors["task_facts_sequence"],
        event_facts_sequence=tensors["event_facts_sequence"],
        padding_mask=tensors["padding_mask"],
        config=config,
    )
    rows = torch.arange(lengths.shape[0], device=lengths.device)
    return targets[rows, lengths - 1].to(dtype=torch.int64).cpu().numpy()


def _cache_metadata(
    context: Any,
    *,
    task: str,
    role: str,
    pool_registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    parent_ids: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": REPRESENTATION_CACHE_SCHEMA,
        "experiment_id": "V27",
        "task": _require_task(task),
        "role": _require_role(role),
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": pool_registry[
            "classification_base_binding_sha256"
        ],
        "pool_registry_sha256": pool_registry["registry_sha256"],
        "open_set_feature_cache_manifest_sha256": feature_manifest[
            "cache_manifest_sha256"
        ],
        "parent_ids": [str(value) for value in parent_ids],
        "parent_ids_sha256": canonical_sha256([str(value) for value in parent_ids]),
        "label_codes": [LABEL_TO_CODE[str(value)] for value in labels],
        "label_names": [str(value) for value in labels],
        "max_sequence_length": int(context.config.data.max_sequence_length),
        "hidden_dim": int(context.config.model.liquid_hidden_dim),
        "summary_feature_dim": int(context.config.model.liquid_hidden_dim * 14),
        "event_trajectory_dim": int(
            context.config.model.liquid_hidden_dim * 7 + 19
        ),
        "task_fact_dim": 8 if task == V27_L1_TASK_NAME else 7,
        "event_fact_dim": 19,
        "unknown_is_model_class": False,
        "unknown_classifier_training_exposure": 0,
        "test_payload_access_count": 0,
    }


def _validate_representation_payload(
    payload: Mapping[str, Any],
    *,
    context: Any,
    task: str,
    role: str,
    pool_registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(payload)
    declared = raw.pop("cache_sha256", None)
    if not isinstance(declared, str) or declared != _representation_cache_digest(payload):
        raise ValueError("V27 representation cache hash cannot be recomputed")
    if raw.get("schema_version") != REPRESENTATION_CACHE_SCHEMA:
        raise ValueError("V27 representation cache schema differs")
    if raw.get("task") != _require_task(task) or raw.get("role") != _require_role(role):
        raise ValueError("V27 representation cache task/role differs")
    if raw.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27 representation cache manifest binding differs")
    if raw.get("pool_registry_sha256") != pool_registry["registry_sha256"]:
        raise ValueError("V27 representation cache pool binding differs")
    if raw.get("open_set_feature_cache_manifest_sha256") != feature_manifest[
        "cache_manifest_sha256"
    ]:
        raise ValueError("V27 representation cache feature binding differs")
    expected_role = pool_registry["roles"][_require_role(role)]
    parent_ids = [str(value) for value in raw.get("parent_ids") or ()]
    if parent_ids != [str(value) for value in expected_role["parent_ids"]]:
        raise ValueError("V27 representation cache parent order differs")
    if raw.get("parent_ids_sha256") != canonical_sha256(parent_ids):
        raise ValueError("V27 representation cache parent hash differs")
    labels = [str(value) for value in raw.get("label_names") or ()]
    if len(labels) != len(parent_ids) or any(value not in LABEL_TO_CODE for value in labels):
        raise ValueError("V27 representation cache labels differ")
    if raw.get("label_codes") != [LABEL_TO_CODE[value] for value in labels]:
        raise ValueError("V27 representation cache label codes differ")
    tensors = raw.pop("tensors", None)
    if not isinstance(tensors, Mapping):
        raise ValueError("V27 representation cache tensors are missing")
    tensors = {str(name): value for name, value in tensors.items()}
    required = {
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
        "padding_mask",
        "lengths",
        "sequence_logits",
    }
    if set(tensors) != required:
        raise ValueError("V27 representation cache tensor fields differ")
    count = len(parent_ids)
    max_length = int(context.config.data.max_sequence_length)
    hidden = int(context.config.model.liquid_hidden_dim)
    task_facts = 8 if task == V27_L1_TASK_NAME else 7
    classes = 2 if task == V27_L1_TASK_NAME else 3
    expected_shapes = {
        "summary_features": (count, max_length, hidden * 14),
        "event_trajectory": (count, max_length, hidden * 7 + 19),
        "task_facts_sequence": (count, max_length, task_facts),
        "event_facts_sequence": (count, max_length, 19),
        "padding_mask": (count, max_length),
        "lengths": (count,),
        "sequence_logits": (count, max_length, classes),
    }
    for name, shape in expected_shapes.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"V27 representation cache tensor shape differs: {name} "
                f"expected={shape}, actual={getattr(value, 'shape', None)}"
            )
        if name == "padding_mask":
            if value.dtype != torch.bool:
                raise ValueError("V27 representation cache padding mask must be bool")
        elif name == "lengths":
            if value.dtype != torch.long:
                raise ValueError("V27 representation cache lengths must be int64")
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"V27 representation cache tensor is non-finite: {name}")
    lengths = tensors["lengths"]
    if bool((lengths < 1).any()) or bool(
        (lengths > max_length).any()
    ):
        raise ValueError("V27 representation cache lengths are out of bounds")
    expected_padding = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(tensors["padding_mask"].cpu(), expected_padding):
        raise ValueError("V27 representation cache padding differs from lengths")
    padding = tensors["padding_mask"]
    for name in (
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
    ):
        if bool(tensors[name].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27 padded representation values are non-zero: {name}")
    if bool(tensors["sequence_logits"].masked_select(padding.unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27 padded representation logits are non-zero")
    return {
        **raw,
        "cache_sha256": declared,
        "tensors": tensors,
    }


def _write_representation_cache(
    context: Any,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    value = dict(payload)
    value["cache_sha256"] = _representation_cache_digest(value)
    if path.exists():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(existing, Mapping):
                existing_digest = existing.get("cache_sha256")
                if (
                    isinstance(existing_digest, str)
                    and existing_digest == value["cache_sha256"]
                    and _representation_cache_digest(existing) == value["cache_sha256"]
                ):
                    return
        except Exception:
            pass
        archive_dir = (
            context.project_root
            / "artifacts"
            / "archive"
            / "v27"
            / "representations"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / f"{path.name}.stale-before-{sha256_file(path)[:16]}"
        if not archived.exists():
            path.replace(archived)
    write_torch_if_absent_or_identical(path, value)


def materialize_representation_caches(context: Any) -> dict[str, Any]:
    """Run frozen L1/L2 bases once and persist causal head inputs."""

    pool_registry, feature_manifest = load_open_set_pool_artifacts(context)
    results: dict[str, Any] = {}
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        device = configured_device(context.config)
        model, checkpoint, binding = load_frozen_base_model(
            context, task=task, device=device
        )
        model.eval()
        for role in OPEN_SET_ROLES:
            role_record = pool_registry["roles"][role]
            cache_paths = open_set_role_cache_paths(
                context, role=role, task=task
            )
            dataset = V27OpenSetDataset(
                context.project_root,
                context.config,
                task=task,
                split_name=role_record["split_name"],
                parent_ids=role_record["parent_ids"],
                **cache_paths,
            )
            loader = build_v27_open_set_dataloader(
                dataset,
                batch_size=context.config.open_set.representation_batch_size,
                shuffle=False,
            )
            max_length = int(context.config.data.max_sequence_length)
            arrays: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "summary_features",
                    "event_trajectory",
                    "task_facts_sequence",
                    "event_facts_sequence",
                    "padding_mask",
                    "lengths",
                    "sequence_logits",
                )
            }
            parent_ids: list[str] = []
            labels: list[str] = []
            with torch.inference_mode():
                for batch in loader:
                    output = model(
                        **model_input_kwargs(batch, device), return_trace=False
                    )
                    lengths = batch["lengths"].to(dtype=torch.long)
                    batch_count = int(lengths.shape[0])
                    fixed_padding = torch.ones(
                        (batch_count, max_length), dtype=torch.bool
                    )
                    fixed_summary = torch.zeros(
                        (
                            batch_count,
                            max_length,
                            model.trajectory_heads.summary_feature_dim,
                        ),
                        dtype=torch.float32,
                    )
                    fixed_event = torch.zeros(
                        (
                            batch_count,
                            max_length,
                            model.trajectory_heads.event_trajectory_dim,
                        ),
                        dtype=torch.float32,
                    )
                    fixed_task = torch.zeros(
                        (batch_count, max_length, model.task_fact_dim),
                        dtype=torch.float32,
                    )
                    fixed_event_facts = torch.zeros(
                        (batch_count, max_length, model.trajectory_event_fact_dim),
                        dtype=torch.float32,
                    )
                    fixed_logits = torch.zeros(
                        (batch_count, max_length, len(model.class_names)),
                        dtype=torch.float32,
                    )
                    for row in range(batch_count):
                        length = int(lengths[row].item())
                        if not 0 < length <= max_length:
                            raise ValueError("V27 representation parent length is invalid")
                        fixed_padding[row, :length] = False
                        fixed_summary[row, :length] = output["summary_features"][
                            row, :length
                        ].detach().cpu()
                        fixed_event[row, :length] = output["event_trajectory"][
                            row, :length
                        ].detach().cpu()
                        fixed_task[row, :length] = output[
                            "task_facts_sequence"
                        ][row, :length].detach().cpu()
                        fixed_event_facts[row, :length] = output[
                            "event_facts_sequence"
                        ][row, :length].detach().cpu()
                        fixed_logits[row, :length] = output["sequence_logits"][
                            row, :length
                        ].detach().cpu()
                    arrays["summary_features"].append(fixed_summary)
                    arrays["event_trajectory"].append(fixed_event)
                    arrays["task_facts_sequence"].append(fixed_task)
                    arrays["event_facts_sequence"].append(fixed_event_facts)
                    arrays["padding_mask"].append(fixed_padding)
                    arrays["lengths"].append(lengths.cpu())
                    arrays["sequence_logits"].append(fixed_logits)
                    parent_ids.extend(str(value) for value in batch["parent_sample_ids"])
                    labels.extend(str(value) for value in batch["base_label_names"])
            if parent_ids != [str(value) for value in role_record["parent_ids"]]:
                raise ValueError("V27 representation cache parent order changed")
            tensors = {
                name: torch.cat(values, dim=0)
                for name, values in arrays.items()
            }
            metadata = _cache_metadata(
                context,
                task=task,
                role=role,
                pool_registry=pool_registry,
                feature_manifest=feature_manifest,
                parent_ids=parent_ids,
                labels=labels,
            )
            payload = {**metadata, "tensors": tensors}
            payload["cache_sha256"] = _representation_cache_digest(payload)
            path = _cache_path(context, task=task, role=role)
            _write_representation_cache(context, path, payload)
            checked = _validate_representation_payload(
                torch.load(path, map_location="cpu", weights_only=False),
                context=context,
                task=task,
                role=role,
                pool_registry=pool_registry,
                feature_manifest=feature_manifest,
            )
            results[f"{task}:{role}"] = {
                "task": task,
                "role": role,
                "parent_count": len(parent_ids),
                "label_counts": dict(Counter(labels)),
                "path": relative_project_path(context.project_root, path),
                "sha256": sha256_file(path),
                "cache_sha256": checked["cache_sha256"],
                "checkpoint_sha256": binding[
                    "l1" if task == V27_L1_TASK_NAME else "l2"
                ]["checkpoint_sha256"],
            }
    manifest_payload: dict[str, Any] = {
        "schema_version": REPRESENTATION_MANIFEST_SCHEMA,
        "experiment_id": "V27",
        "status": "prepared",
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": pool_registry[
            "classification_base_binding_sha256"
        ],
        "pool_registry_sha256": pool_registry["registry_sha256"],
        "open_set_feature_cache_manifest_sha256": feature_manifest[
            "cache_manifest_sha256"
        ],
        "tasks": results,
        "test_payload_access_count": 0,
        "unknown_classifier_training_exposure": 0,
    }
    manifest_payload["representation_cache_manifest_sha256"] = canonical_sha256(
        manifest_payload
    )
    manifest_path = resolve_project_path(
        context.project_root,
        context.config.artifacts.representation_cache_manifest_path,
    )
    write_v27_sealed_json(manifest_path, manifest_payload)
    return {
        "status": "prepared",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "representation_cache_manifest_sha256": manifest_payload[
            "representation_cache_manifest_sha256"
        ],
        "tasks": results,
        "test_payload_access_count": 0,
    }


def load_representation_cache_manifest(context: Any) -> dict[str, Any]:
    pool_registry, feature_manifest = load_open_set_pool_artifacts(context)
    path = resolve_project_path(
        context.project_root,
        context.config.artifacts.representation_cache_manifest_path,
        must_exist=True,
    )
    raw = read_v27_sealed_json(path)
    payload = dict(raw)
    sealed = payload.pop("payload_sha256", None)
    declared = payload.pop("representation_cache_manifest_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 representation manifest hash cannot be recomputed")
    if sealed != canonical_sha256(
        {**payload, "representation_cache_manifest_sha256": declared}
    ):
        raise ValueError("V27 representation manifest payload hash differs")
    if (
        payload.get("schema_version") != REPRESENTATION_MANIFEST_SCHEMA
        or payload.get("status") != "prepared"
        or payload.get("system_manifest_sha256") != context.manifest_sha256
        or payload.get("pool_registry_sha256") != pool_registry["registry_sha256"]
        or payload.get("open_set_feature_cache_manifest_sha256")
        != feature_manifest["cache_manifest_sha256"]
    ):
        raise ValueError("V27 representation manifest binding differs")
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != {
        f"{task}:{role}" for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME) for role in OPEN_SET_ROLES
    }:
        raise ValueError("V27 representation manifest task set differs")
    result = {
        **payload,
        "representation_cache_manifest_sha256": declared,
        "payload_sha256": sealed,
    }
    for key, entry in tasks.items():
        task, role = key.split(":", 1)
        path_value = resolve_project_path(
            context.project_root, str(entry.get("path", "")), must_exist=True
        )
        if sha256_file(path_value) != entry.get("sha256"):
            raise ValueError(f"V27 representation cache file hash differs: {key}")
        checked = load_representation_cache(context, task=task, role=role)
        if checked["cache_sha256"] != entry.get("cache_sha256"):
            raise ValueError(f"V27 representation cache digest differs: {key}")
    return result


def load_representation_cache(context: Any, *, task: str, role: str) -> dict[str, Any]:
    task = _require_task(task)
    role = _require_role(role)
    pool_registry, feature_manifest = load_open_set_pool_artifacts(context)
    path = _cache_path(context, task=task, role=role)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27 representation cache is not an object")
    return _validate_representation_payload(
        payload,
        context=context,
        task=task,
        role=role,
        pool_registry=pool_registry,
        feature_manifest=feature_manifest,
    )


class _CachedTrajectoryDataset(Dataset[dict[str, Any]]):
    """Parent-level view over one verified frozen representation cache."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        tensors = payload["tensors"]
        self.tensors = {str(name): value for name, value in tensors.items()}
        self.labels = [str(value) for value in payload["label_names"]]
        self.parent_ids = [str(value) for value in payload["parent_ids"]]
        if len(self.labels) != len(self.parent_ids):
            raise ValueError("V27 cached dataset metadata lengths differ")

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("V27 cached dataset index must be an integer")
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError("V27 cached dataset index is out of range")
        return {
            "summary_features": self.tensors["summary_features"][position],
            "event_trajectory": self.tensors["event_trajectory"][position],
            "task_facts_sequence": self.tensors["task_facts_sequence"][position],
            "event_facts_sequence": self.tensors["event_facts_sequence"][position],
            "padding_mask": self.tensors["padding_mask"][position],
            "lengths": self.tensors["lengths"][position],
            "sequence_logits": self.tensors["sequence_logits"][position],
            "label_name": self.labels[position],
            "parent_sample_id": self.parent_ids[position],
        }


def _collate_cached(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("V27 cached head batch cannot be empty")
    tensor_names = (
        "summary_features",
        "event_trajectory",
        "task_facts_sequence",
        "event_facts_sequence",
        "padding_mask",
        "lengths",
        "sequence_logits",
    )
    return {
        **{name: torch.stack([item[name] for item in items], dim=0) for name in tensor_names},
        "label_names": [item["label_name"] for item in items],
        "parent_sample_ids": [item["parent_sample_id"] for item in items],
    }


def _knownness_targets(task: str, labels: Sequence[str]) -> torch.Tensor:
    task = _require_task(task)
    if task == V27_L1_TASK_NAME:
        return torch.tensor(
            [0.0 if str(label) == "UnknownTarget" else 1.0 for label in labels],
            dtype=torch.float32,
        )
    return torch.tensor(
        [1.0 if str(label) in {"DroneTarget", "BirdTarget", "BalloonTarget"} else 0.0 for label in labels],
        dtype=torch.float32,
    )


def deterministic_quality_targets(
    task: str,
    *,
    task_facts_sequence: torch.Tensor,
    event_facts_sequence: torch.Tensor,
    padding_mask: torch.Tensor,
    config: V27Config,
) -> torch.Tensor:
    """Build an auditable causal maturity target from facts only.

    It intentionally does not inspect labels, classifier predictions or
    learned scores.  The target is therefore a quality/maturity contract, not
    a disguised Unknown label.
    """

    task = _require_task(task)
    if task_facts_sequence.ndim != 3 or event_facts_sequence.ndim != 3:
        raise ValueError("V27 quality facts must be batched sequences")
    if padding_mask.shape != task_facts_sequence.shape[:2] or event_facts_sequence.shape[:2] != padding_mask.shape:
        raise ValueError("V27 quality facts and padding shapes differ")
    if task_facts_sequence.shape[-1] != (8 if task == V27_L1_TASK_NAME else 7) or event_facts_sequence.shape[-1] != 19:
        raise ValueError("V27 quality fact dimensions differ")
    active = ~padding_mask
    event_number = torch.arange(
        1, padding_mask.shape[1] + 1, device=padding_mask.device
    ).unsqueeze(0)
    rd_fraction = task_facts_sequence[..., 1]
    dt_fraction = task_facts_sequence[..., 3]
    if task == V27_L1_TASK_NAME:
        aux_fraction = task_facts_sequence[..., 7]
    else:
        aux_fraction = torch.minimum(
            task_facts_sequence[..., 5], task_facts_sequence[..., 6]
        )
    target = (
        active
        & (event_number >= int(config.open_set.quality_min_events))
        & (rd_fraction >= float(config.open_set.quality_min_rd_fraction))
        & (dt_fraction >= float(config.open_set.quality_min_dt_fraction))
        & (aux_fraction >= float(config.open_set.quality_min_aux_fraction))
    )
    return target


def _eligible_prefix_mask(
    padding_mask: torch.Tensor, *, minimum_events: int
) -> torch.Tensor:
    event_number = torch.arange(
        1, padding_mask.shape[1] + 1, device=padding_mask.device
    ).unsqueeze(0)
    return (~padding_mask) & (event_number >= int(minimum_events))


def _parent_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("V27 parent-balanced BCE expects [B,T] tensors")
    values = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(float(positive_weight)),
        reduction="none",
    )
    active_count = mask.sum(dim=1)
    eligible = active_count > 0
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    means = (values * mask.to(dtype=values.dtype)).sum(dim=1) / active_count.clamp_min(1)
    return means[eligible].mean()


def _positive_weight(targets: torch.Tensor, mask: torch.Tensor) -> float:
    selected = targets.masked_select(mask)
    positives = int(selected.sum().item())
    negatives = int(selected.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError(
            f"V27 head target must contain both classes: positives={positives}, negatives={negatives}"
        )
    return max(1.0e-3, float(negatives) / float(positives))


def _head_state(model: V27LiquidV2TaskModel) -> dict[str, torch.Tensor]:
    keys = mutable_trajectory_head_keys(model.state_dict())
    prefix = "trajectory_heads."
    return {
        key[len(prefix) :]: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key in keys
    }


def _load_head_state(model: V27LiquidV2TaskModel, state: Mapping[str, Any]) -> None:
    expected = {
        key[len("trajectory_heads.") :]: value
        for key, value in model.state_dict().items()
        if key.startswith("trajectory_heads.")
    }
    if set(state) != set(expected):
        raise ValueError("V27 open-set head state keys differ")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"V27 open-set head state shape/dtype differs: {key}")
    model.trajectory_heads.load_state_dict(dict(state), strict=True)


def _head_checkpoint_payload(
    context: Any,
    *,
    task: str,
    seed: int,
    stage: str,
    model: V27LiquidV2TaskModel,
    metrics: Mapping[str, Any],
    representation_manifest: Mapping[str, Any],
    base_record: Mapping[str, Any],
    global_step: int,
) -> dict[str, Any]:
    state = _head_state(model)
    payload: dict[str, Any] = {
        "checkpoint_schema": HEAD_CHECKPOINT_SCHEMA,
        "experiment_id": "V27",
        "task": task,
        "stage": stage,
        "seed": int(seed),
        "epoch": int(metrics.get("epoch", 0)),
        "global_step": int(global_step),
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": representation_manifest[
            "frozen_base_binding_sha256"
        ],
        "base_checkpoint_sha256": base_record["checkpoint_sha256"],
        "base_classifier_state_sha256": base_record["classifier_state_sha256"],
        "base_model_contract_sha256": base_record["model_contract_sha256"],
        "representation_cache_manifest_sha256": representation_manifest[
            "representation_cache_manifest_sha256"
        ],
        "model_contract": model.model_contract(),
        "model_contract_sha256": canonical_sha256(model.model_contract()),
        "trajectory_head_state_sha256": _tensor_digest(state),
        "metrics": dict(metrics),
        "stage_contract": _head_stage_contract(stage),
        "head_state": state,
        "unknown_is_model_class": False,
        "unknown_classifier_training_exposure": 0,
        "deployment_eligible": False,
    }
    return payload


def _verify_head_checkpoint(
    payload: Mapping[str, Any],
    context: Any,
    *,
    task: str,
    representation_manifest: Mapping[str, Any],
    base_record: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(payload)
    if value.get("checkpoint_schema") != HEAD_CHECKPOINT_SCHEMA or value.get("task") != task:
        raise ValueError("V27 open-set head checkpoint schema/task differs")
    if value.get("seed") not in context.config.training.seeds:
        raise ValueError("V27 open-set head checkpoint seed differs")
    if value.get("system_manifest_sha256") != context.manifest_sha256 or value.get(
        "frozen_base_binding_sha256"
    ) != representation_manifest["frozen_base_binding_sha256"]:
        raise ValueError("V27 open-set head checkpoint manifest binding differs")
    if value.get("base_checkpoint_sha256") != base_record["checkpoint_sha256"] or value.get(
        "base_classifier_state_sha256"
    ) != base_record["classifier_state_sha256"]:
        raise ValueError("V27 open-set head checkpoint base binding differs")
    if value.get("representation_cache_manifest_sha256") != representation_manifest[
        "representation_cache_manifest_sha256"
    ]:
        raise ValueError("V27 open-set head checkpoint representation binding differs")
    contract = value.get("model_contract")
    if not isinstance(contract, Mapping) or value.get("model_contract_sha256") != canonical_sha256(dict(contract)):
        raise ValueError("V27 open-set head model contract hash differs")
    expected_model = V27LiquidV2TaskModel(context.config, task=task)
    expected_contract = expected_model.model_contract()
    if dict(contract) != expected_contract:
        raise ValueError("V27 open-set head model contract differs from current model")
    try:
        expected_stage_contract = _head_stage_contract(str(value.get("stage")))
    except ValueError as exc:
        raise ValueError("V27 open-set head checkpoint stage is invalid") from exc
    if value.get("stage_contract") != expected_stage_contract:
        raise ValueError("V27 open-set head stage contract differs")
    state = value.get("head_state")
    if not isinstance(state, Mapping) or value.get("trajectory_head_state_sha256") != _tensor_digest(state):
        raise ValueError("V27 open-set head state hash differs")
    if value.get("stage") == "final":
        provenance = value.get("component_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("V27 final head component provenance is missing")
        for component in ("knownness", "quality"):
            if provenance.get(f"{component}_component_state_sha256") != _head_component_state_hash(
                state, component=component
            ):
                raise ValueError(
                    f"V27 final head {component} component provenance differs"
                )
            digest = provenance.get(f"{component}_checkpoint_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"V27 final head {component} checkpoint digest differs"
                )
    return value


def _score_summary(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size != s.size or y.size == 0 or not np.isfinite(s).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("V27 score summary requires finite binary support")
    positives = s[y == 1]
    negatives = s[y == 0]
    # Mann-Whitney AUC with average ties; avoids adding a metric dependency.
    values = np.concatenate((positives, negatives))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    auc = (ranks[: positives.size].sum() - positives.size * (positives.size + 1) / 2.0) / (
        positives.size * negatives.size
    )
    return {
        "support": int(y.size),
        "positive_support": int(positives.size),
        "negative_support": int(negatives.size),
        "auroc": float(auc),
        "positive_mean": float(positives.mean()),
        "negative_mean": float(negatives.mean()),
    }


def _select_high_score_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    recall_floor: float,
) -> tuple[float, dict[str, Any]]:
    positives = np.asarray(positive_scores, dtype=np.float64)
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if positives.size == 0 or negatives.size == 0 or not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError("V27 threshold fitting requires finite positive and negative scores")
    floor = float(recall_floor)
    if not 0.0 < floor <= 1.0:
        raise ValueError("V27 threshold recall floor must be in (0,1]")
    candidates = np.unique(np.concatenate((positives, negatives)))
    feasible: list[tuple[float, float, float, int, int]] = []
    for threshold in candidates:
        tp = int(np.count_nonzero(positives >= threshold))
        fn = int(positives.size - tp)
        fp = int(np.count_nonzero(negatives >= threshold))
        recall = tp / positives.size
        if recall + 1.0e-12 >= floor:
            feasible.append((float(threshold), recall, 1.0 - fp / negatives.size, tp, fp))
    if not feasible:
        threshold = float(np.nextafter(np.min(positives), -np.inf))
        tp = int(np.count_nonzero(positives >= threshold))
        fp = int(np.count_nonzero(negatives >= threshold))
        return threshold, {
            "recall": tp / positives.size,
            "negative_rejection": 1.0 - fp / negatives.size,
            "true_positive": tp,
            "false_positive": fp,
            "feasible": False,
        }
    # First maximise negative rejection, then threshold.  Recall floor is a
    # safety constraint, not an objective that may be traded away.
    chosen = max(feasible, key=lambda row: (row[2], row[0]))
    threshold, recall, rejection, tp, fp = chosen
    return threshold, {
        "recall": recall,
        "negative_rejection": rejection,
        "true_positive": tp,
        "false_positive": fp,
        "feasible": True,
    }


def _fit_temperature(logits: np.ndarray, labels: np.ndarray, *, lower: float, upper: float) -> float:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        raise ValueError("V27 temperature fitting inputs have invalid shapes")
    if not np.isfinite(logits).all() or not np.isfinite(labels).all() or not lower <= upper:
        raise ValueError("V27 temperature fitting inputs are invalid")
    tensor_logits = torch.from_numpy(logits.astype(np.float64))
    tensor_labels = torch.from_numpy(labels.astype(np.int64))

    def objective(value: float) -> float:
        if value <= 0.0:
            return float("inf")
        return float(
            nn.functional.cross_entropy(
                tensor_logits / float(value), tensor_labels
            ).item()
        )

    # A deterministic logarithmic grid is robust to tiny calibration pools and
    # does not make scipy an execution prerequisite.
    grid = np.geomspace(float(lower), float(upper), num=161)
    values = np.asarray([objective(float(value)) for value in grid])
    best = int(np.argmin(values))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]
    for _ in range(32):
        first = left + (right - left) / 3.0
        second = right - (right - left) / 3.0
        if objective(float(first)) <= objective(float(second)):
            right = second
        else:
            left = first
    candidate = (left + right) / 2.0
    return float(np.clip(candidate, lower, upper))


def _calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("V27 calibrated probabilities are non-finite")
    return probabilities


def _with_head_scores(
    context: Any,
    *,
    task: str,
    role: str,
    head_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a cached representation through one fitted trajectory head."""

    cache = load_representation_cache(context, task=task, role=role)
    model, _, binding = load_frozen_base_model(
        context,
        task=task,
        device=torch.device("cpu"),
    )
    _load_head_state(model, head_payload["head_state"])
    model.eval()
    tensors = cache["tensors"]
    dataset = _CachedTrajectoryDataset(cache)
    loader = DataLoader(
        dataset,
        batch_size=context.config.open_set.head_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_cached,
    )
    knownness: list[torch.Tensor] = []
    quality: list[torch.Tensor] = []
    order: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            output = model.trajectory_heads.forward_from_cached_features(
                summary_features=batch["summary_features"],
                event_trajectory=batch["event_trajectory"],
                task_facts_sequence=batch["task_facts_sequence"],
                event_facts_sequence=batch["event_facts_sequence"],
                padding_mask=batch["padding_mask"],
            )
            knownness.append(output["knownness_logit_sequence"].cpu())
            quality.append(output["quality_logit_sequence"].cpu())
            order.extend(batch["parent_sample_ids"])
    expected = [str(value) for value in cache["parent_ids"]]
    if order != expected:
        raise ValueError("V27 fitted head replay changed parent order")
    enriched = dict(cache)
    enriched["tensors"] = {
        **tensors,
        "knownness_logit": torch.cat(knownness, dim=0),
        "quality_logit": torch.cat(quality, dim=0),
    }
    enriched["tensors"]["knownness_logit"] = enriched["tensors"]["knownness_logit"].masked_fill(
        enriched["tensors"]["padding_mask"], 0.0
    )
    enriched["tensors"]["quality_logit"] = enriched["tensors"]["quality_logit"].masked_fill(
        enriched["tensors"]["padding_mask"], 0.0
    )
    enriched["head_state_sha256"] = head_payload["trajectory_head_state_sha256"]
    enriched["seed"] = int(head_payload["seed"])
    enriched["task"] = task
    enriched["role"] = role
    enriched["base_checkpoint_sha256"] = binding[
        "l1" if task == V27_L1_TASK_NAME else "l2"
    ]["checkpoint_sha256"]
    # Preserve the immutable representation digest as the parent binding.
    # Derived score artifacts have their own digest and must never replace it.
    enriched["representation_cache_sha256"] = cache["cache_sha256"]
    return enriched


def _write_head_score_cache(
    context: Any,
    *,
    task: str,
    role: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(payload)
    value["score_cache_schema"] = "RADAR-LN3-V27-fitted-head-score-cache-v1"
    representation_digest = value.pop("cache_sha256", None)
    if not isinstance(representation_digest, str):
        representation_digest = value.get("representation_cache_sha256")
    if not isinstance(representation_digest, str):
        raise ValueError("V27 score cache lacks representation cache binding")
    value["representation_cache_sha256"] = representation_digest
    value["score_cache_sha256"] = _score_cache_digest(value)
    score_path = _score_cache_path(
        context,
        task=task,
        role=role,
        seed=int(value.get("seed", context.config.training.seeds[0])),
    )
    if score_path.exists():
        try:
            existing = torch.load(score_path, map_location="cpu", weights_only=False)
            existing_current = (
                isinstance(existing, Mapping)
                and existing.get("score_cache_sha256") == value["score_cache_sha256"]
                and _score_cache_digest(existing) == value["score_cache_sha256"]
            )
        except Exception:
            existing_current = False
        if not existing_current:
            _archive_derived_artifact(
                context, score_path, category="score-caches"
            )
    write_torch_if_absent_or_identical(score_path, value)
    return value


def load_head_score_cache(
    context: Any, *, task: str, role: str, seed: int | None = None
) -> dict[str, Any]:
    actual_seed = int(
        context.config.training.seeds[0] if seed is None else seed
    )
    path = _score_cache_path(
        context, task=task, role=role, seed=actual_seed
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("score_cache_schema") != "RADAR-LN3-V27-fitted-head-score-cache-v1":
        raise ValueError("V27 fitted head score cache schema differs")
    declared = payload.get("score_cache_sha256")
    if not isinstance(declared, str) or declared != _score_cache_digest(payload):
        raise ValueError("V27 fitted head score cache hash differs")
    base = load_representation_cache(context, task=task, role=role)
    if (
        payload.get("representation_cache_sha256") != base["cache_sha256"]
        or payload.get("head_state_sha256") is None
        or payload.get("task") != task
        or payload.get("role") != role
        or int(payload.get("seed", -1)) != actual_seed
    ):
        raise ValueError("V27 fitted head score cache representation binding differs")
    binding = load_frozen_base_model(
        context, task=task, device=torch.device("cpu")
    )[2]
    expected_checkpoint = binding[
        "l1" if task == V27_L1_TASK_NAME else "l2"
    ]["checkpoint_sha256"]
    if payload.get("base_checkpoint_sha256") != expected_checkpoint:
        raise ValueError("V27 fitted head score cache base binding differs")
    if (
        [str(value) for value in payload.get("parent_ids") or ()]
        != [str(value) for value in base["parent_ids"]]
        or [str(value) for value in payload.get("label_names") or ()]
        != [str(value) for value in base["label_names"]]
    ):
        raise ValueError("V27 fitted head score cache parent/label binding differs")
    score_tensors = payload.get("tensors")
    if not isinstance(score_tensors, Mapping):
        raise ValueError("V27 fitted head score cache tensors are missing")
    for name, base_value in base["tensors"].items():
        cached_value = score_tensors.get(name)
        if (
            not isinstance(cached_value, torch.Tensor)
            or cached_value.dtype != base_value.dtype
            or cached_value.shape != base_value.shape
            or not torch.equal(cached_value.detach().cpu(), base_value.detach().cpu())
        ):
            raise ValueError(
                f"V27 fitted head score cache copied representation differs: {name}"
            )
    for name in ("knownness_logit", "quality_logit"):
        value = score_tensors.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != base["tensors"]["padding_mask"].shape or not torch.isfinite(value).all():
            raise ValueError(f"V27 fitted head score field differs: {name}")
    return dict(payload)


def materialize_head_score_cache(
    context: Any,
    *,
    task: str,
    role: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Replay one finalized head over one frozen representation role.

    Score caches are derived artifacts, not training inputs.  Replays with the
    same finalized head and representation cache are reused; a changed parent
    binding is archived before a new derived cache is written.
    """

    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    _require_task(task)
    _require_role(role)
    head_payload, _, _ = _load_final_head(
        context, task=task, seed=actual_seed
    )
    payload = _with_head_scores(
        context,
        task=task,
        role=role,
        head_payload=head_payload,
    )
    cached = _write_head_score_cache(
        context,
        task=task,
        role=role,
        payload=payload,
    )
    if cached.get("head_state_sha256") != head_payload[
        "trajectory_head_state_sha256"
    ]:
        raise ValueError("V27 materialized head score state binding differs")
    return load_head_score_cache(
        context,
        task=task,
        role=role,
        seed=actual_seed,
    )


def _fit_head_component(
    context: Any,
    *,
    task: str,
    seed: int,
    component: str,
    initial_head_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one independent head component on representation_fit."""

    task = _require_task(task)
    if component not in {"knownness", "quality"}:
        raise ValueError("V27 head component must be knownness or quality")
    representation_manifest = load_representation_cache_manifest(context)
    _archive_stale_head_stage_artifacts(
        context,
        task=task,
        stage=component,
        seed=seed,
        representation_manifest=representation_manifest,
    )
    model, _, frozen_binding = load_frozen_base_model(
        context,
        task=task,
        device=configured_device(context.config),
    )
    base_record = frozen_binding["l1" if task == V27_L1_TASK_NAME else "l2"]
    if initial_head_payload is not None:
        _verify_head_checkpoint(
            initial_head_payload,
            context,
            task=task,
            representation_manifest=representation_manifest,
            base_record=base_record,
        )
        _load_head_state(model, initial_head_payload["head_state"])
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if component == "knownness":
        trainable = list(model.trajectory_heads.trajectory_encoder.parameters()) + list(
            model.trajectory_heads.knownness_projection.parameters()
        ) + list(model.trajectory_heads.knownness_head.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
    else:
        trainable = list(model.trajectory_heads.quality_head.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
    if not trainable:
        raise AssertionError("V27 open-set head stage has no trainable parameters")
    dataset_payload = load_representation_cache(
        context, task=task, role="representation_fit"
    )
    known_weight, quality_weight = _head_target_weights(
        dataset_payload, task=task, config=context.config
    )
    dataset = _CachedTrajectoryDataset(dataset_payload)
    loader = DataLoader(
        dataset,
        batch_size=context.config.open_set.head_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_cached,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=context.config.open_set.head_learning_rate,
        weight_decay=context.config.open_set.head_weight_decay,
    )
    logger = configure_training_logger(
        _head_log_path(context, task=task, component=component, seed=seed),
        logger_name=(
            f"{TRAINING_LOGGER_PREFIX}.{task}.open_set.{component}.seed_{seed}"
        ),
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    global_step = 0
    try:
        log_event(
            logger,
            "open_set_head_run=start",
            experiment="V27",
            task=task,
            component=component,
            seed=seed,
            train_parents=len(dataset),
            epochs=context.config.open_set.head_epochs,
            batch_size=context.config.open_set.head_batch_size,
            frozen_base_sha256=base_record["checkpoint_sha256"],
            representation_manifest_sha256=representation_manifest[
                "representation_cache_manifest_sha256"
            ],
        )
        for epoch in range(1, context.config.open_set.head_epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                device = _device_of(model)
                summary_features = batch["summary_features"].to(device)
                output = model.trajectory_heads.forward_from_cached_features(
                    summary_features=summary_features,
                    event_trajectory=batch["event_trajectory"].to(device),
                    task_facts_sequence=batch["task_facts_sequence"].to(device),
                    event_facts_sequence=batch["event_facts_sequence"].to(device),
                    padding_mask=batch["padding_mask"].to(device),
                )
                labels = batch["label_names"]
                known_parent = _knownness_targets(task, labels).to(device)
                known_target = known_parent.unsqueeze(1).expand_as(output["knownness_logit_sequence"])
                padding = batch["padding_mask"].to(device)
                known_mask = _eligible_prefix_mask(
                    padding, minimum_events=context.config.open_set.knownness_min_events
                )
                quality_target = deterministic_quality_targets(
                    task,
                    task_facts_sequence=batch["task_facts_sequence"].to(device),
                    event_facts_sequence=batch["event_facts_sequence"].to(device),
                    padding_mask=padding,
                    config=context.config,
                ).to(dtype=torch.float32)
                quality_mask = ~padding
                if component == "knownness":
                    loss = _parent_balanced_bce(
                        output["knownness_logit_sequence"],
                        known_target,
                        known_mask,
                        positive_weight=known_weight,
                    )
                else:
                    loss = _parent_balanced_bce(
                        output["quality_logit_sequence"],
                        quality_target,
                        quality_mask,
                        positive_weight=quality_weight,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V27 open-set head loss is non-finite")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    context.config.open_set.head_gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().item()))
                if global_step == 1 or global_step % progress_interval(max(1, len(loader))) == 0:
                    log_event(
                        logger,
                        "open_set_head_progress",
                        component=component,
                        epoch=f"{epoch:02d}/{context.config.open_set.head_epochs:02d}",
                        global_step=global_step,
                        loss=float(loss.detach().item()),
                        gradient_norm=float(gradient_norm.item()),
                    )
            development = _evaluate_head_epoch(
                context, model, task=task, role="representation_development"
            )
            score = float(
                development["knownness"]["auroc"]
                if component == "knownness"
                else development["quality"]["auroc"]
            )
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "development": development,
                "selection_score": score,
                "component": component,
            }
            history.append(record)
            if score > best_score or best_state is None:
                best_score = score
                best_state = _head_state(model)
                best_epoch = epoch
            log_event(
                logger,
                "open_set_head_epoch=complete",
                component=component,
                epoch=epoch,
                train_loss=record["train_loss"],
                development_auroc=score,
                best_score=best_score,
                elapsed=format_duration(time.perf_counter() - started),
            )
        if best_state is None or best_epoch is None:
            raise RuntimeError("V27 open-set head stage produced no checkpoint")
        _load_head_state(model, best_state)
        metrics = {
            "epoch": int(best_epoch),
            "component": component,
            "best_development_score": best_score,
            "history_length": len(history),
        }
        payload = _head_checkpoint_payload(
            context,
            task=task,
            seed=seed,
            stage=component,
            model=model,
            metrics=metrics,
            representation_manifest=representation_manifest,
            base_record=base_record,
            global_step=global_step,
        )
        checkpoint_path = _head_stage_path(
            context, task=task, stage=component, seed=seed
        )
        write_torch_if_absent_or_identical(checkpoint_path, payload)
        summary = {
            "status": "completed_open_set_head_component",
            "schema_version": HEAD_SUMMARY_SCHEMA,
            "experiment_id": "V27",
            "task": task,
            "component": component,
            "seed": seed,
            "history": history,
            "best_development_score": best_score,
            "selected_epoch": int(best_epoch),
            "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "representation_manifest_sha256": representation_manifest[
                "representation_cache_manifest_sha256"
            ],
            "frozen_base_checkpoint_sha256": base_record["checkpoint_sha256"],
            "environment": environment_snapshot(),
            "training_log_path": relative_project_path(
                context.project_root,
                _head_log_path(
                    context, task=task, component=component, seed=seed
                ),
            ),
            "unknown_classifier_training_exposure": 0,
            "test_payload_access_count": 0,
            "deployment_eligible": False,
        }
        summary_path = _head_stage_summary_path(
            context, task=task, stage=component, seed=seed
        )
        _write_derived_sealed_json(context, summary_path, summary)
        log_event(
            logger,
            "open_set_head_run=complete",
            task=task,
            component=component,
            checkpoint_path=str(checkpoint_path),
            summary_path=str(summary_path),
            elapsed=format_duration(time.perf_counter() - started),
        )
        return {**summary, "summary_path": str(summary_path)}
    except Exception:
        logger.exception(
            "open_set_head_run=failed | task=%s | component=%s | elapsed=%s",
            task,
            component,
            format_duration(time.perf_counter() - started),
        )
        raise
    finally:
        close_training_logger(logger)


def _evaluate_head_epoch(
    context: Any,
    model: V27LiquidV2TaskModel,
    *,
    task: str,
    role: str,
) -> dict[str, Any]:
    payload = load_representation_cache(context, task=task, role=role)
    model.eval()
    device = _device_of(model)
    dataset = _CachedTrajectoryDataset(payload)
    loader = DataLoader(dataset, batch_size=context.config.open_set.head_batch_size, shuffle=False, collate_fn=_collate_cached)
    known_scores: list[float] = []
    known_labels: list[int] = []
    quality_scores: list[float] = []
    quality_labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model.trajectory_heads.forward_from_cached_features(
                summary_features=batch["summary_features"],
                event_trajectory=batch["event_trajectory"],
                task_facts_sequence=batch["task_facts_sequence"],
                event_facts_sequence=batch["event_facts_sequence"],
                padding_mask=batch["padding_mask"],
            )
            lengths = batch["lengths"]
            rows = torch.arange(lengths.shape[0], device=device)
            final = lengths - 1
            known_scores.extend(
                torch.sigmoid(output["knownness_logit_sequence"][rows, final])
                .detach()
                .cpu()
                .tolist()
            )
            known_labels.extend(
                _knownness_targets(task, batch["label_names"])
                .to(dtype=torch.long)
                .detach()
                .cpu()
                .tolist()
            )
            quality_target = deterministic_quality_targets(
                task,
                task_facts_sequence=batch["task_facts_sequence"],
                event_facts_sequence=batch["event_facts_sequence"],
                padding_mask=batch["padding_mask"],
                config=context.config,
            )
            quality_scores.extend(
                torch.sigmoid(output["quality_logit_sequence"][rows, final])
                .detach()
                .cpu()
                .tolist()
            )
            quality_labels.extend(
                quality_target[rows, final]
                .to(dtype=torch.long)
                .detach()
                .cpu()
                .tolist()
            )
    return {
        "knownness": _score_summary(known_labels, known_scores),
        "quality": _score_summary(quality_labels, quality_scores),
    }


def train_open_set_heads(
    context: Any,
    *,
    task: str,
    component: str,
    seed: int | None = None,
) -> dict[str, Any]:
    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    if actual_seed not in context.config.training.seeds:
        raise ValueError("V27 open-set head seed is not registered")
    seed_v27(actual_seed)
    initial = None
    if component == "quality":
        knownness_path = _head_stage_path(
            context, task=task, stage="knownness", seed=actual_seed
        )
        if not knownness_path.is_file():
            raise FileNotFoundError(
                "V27 quality head training requires the knownness checkpoint first: "
                f"{knownness_path}"
            )
        initial = torch.load(knownness_path, map_location="cpu", weights_only=False)
    return _fit_head_component(
        context,
        task=_require_task(task),
        seed=actual_seed,
        component=component,
        initial_head_payload=initial,
    )


def finalize_open_set_heads(
    context: Any,
    *,
    task: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Copy the independently fitted quality stage to the final head path."""

    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    quality_path = _head_stage_path(
        context, task=task, stage="quality", seed=actual_seed
    )
    if not quality_path.is_file():
        raise FileNotFoundError("V27 quality head checkpoint is missing")
    payload = torch.load(quality_path, map_location="cpu", weights_only=False)
    representation_manifest = load_representation_cache_manifest(context)
    binding = load_frozen_base_model(
        context, task=task, device=torch.device("cpu")
    )[2]
    base_record = binding["l1" if task == V27_L1_TASK_NAME else "l2"]
    _verify_head_checkpoint(
        payload,
        context,
        task=task,
        representation_manifest=representation_manifest,
        base_record=base_record,
    )
    knownness_path = _head_stage_path(
        context, task=task, stage="knownness", seed=actual_seed
    )
    if not knownness_path.is_file():
        raise FileNotFoundError("V27 knownness head checkpoint is missing")
    knownness_payload = torch.load(
        knownness_path, map_location="cpu", weights_only=False
    )
    _verify_head_checkpoint(
        knownness_payload,
        context,
        task=task,
        representation_manifest=representation_manifest,
        base_record=base_record,
    )
    if _head_component_state_hash(
        payload["head_state"], component="knownness"
    ) != _head_component_state_hash(
        knownness_payload["head_state"], component="knownness"
    ):
        raise ValueError(
            "V27 quality stage changed the independently fitted knownness state"
        )
    model, _, _ = load_frozen_base_model(context, task=task, device=torch.device("cpu"))
    _load_head_state(model, payload["head_state"])
    final_payload = dict(payload)
    final_payload["stage"] = "final"
    final_payload["checkpoint_schema"] = HEAD_CHECKPOINT_SCHEMA
    final_payload["stage_contract"] = _head_stage_contract("final")
    final_payload["component_provenance"] = {
        "knownness_checkpoint_sha256": sha256_file(knownness_path),
        "quality_checkpoint_sha256": sha256_file(quality_path),
        "knownness_component_state_sha256": _head_component_state_hash(
            payload["head_state"], component="knownness"
        ),
        "quality_component_state_sha256": _head_component_state_hash(
            payload["head_state"], component="quality"
        ),
    }
    final_payload["metrics"] = {
        **dict(payload.get("metrics") or {}),
        "finalized_from_stage": "quality",
    }
    final_payload["model_contract"] = model.model_contract()
    final_payload["model_contract_sha256"] = canonical_sha256(
        final_payload["model_contract"]
    )
    final_path = _head_stage_path(context, task=task, stage="final", seed=actual_seed)
    if final_path.exists():
        try:
            existing = torch.load(final_path, map_location="cpu", weights_only=False)
            existing_current = (
                isinstance(existing, Mapping)
                and existing.get("trajectory_head_state_sha256")
                == final_payload["trajectory_head_state_sha256"]
                and existing.get("system_manifest_sha256")
                == context.manifest_sha256
            )
        except Exception:
            existing_current = False
        if not existing_current:
            _archive_derived_artifact(context, final_path, category="head-stages")
    write_torch_if_absent_or_identical(final_path, final_payload)
    return {
        "status": "finalized",
        "task": task,
        "checkpoint_path": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "source_checkpoint_sha256": sha256_file(quality_path),
    }


def _load_final_head(context: Any, *, task: str, seed: int) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    representation_manifest = load_representation_cache_manifest(context)
    binding = load_frozen_base_model(context, task=task, device=torch.device("cpu"))[2]
    base_record = binding["l1" if task == V27_L1_TASK_NAME else "l2"]
    path = _head_stage_path(context, task=task, stage="final", seed=seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checked = _verify_head_checkpoint(
        payload,
        context,
        task=task,
        representation_manifest=representation_manifest,
        base_record=base_record,
    )
    provenance = checked.get("component_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V27 final head component provenance is missing")
    source_payloads: dict[str, Mapping[str, Any]] = {}
    for component in ("knownness", "quality"):
        source_path = _head_stage_path(
            context, task=task, stage=component, seed=seed
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"V27 final head source checkpoint is missing: {source_path}"
            )
        if sha256_file(source_path) != provenance.get(
            f"{component}_checkpoint_sha256"
        ):
            raise ValueError(
                f"V27 final head {component} source checkpoint hash differs"
            )
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        source_payloads[component] = _verify_head_checkpoint(
            source,
            context,
            task=task,
            representation_manifest=representation_manifest,
            base_record=base_record,
        )
        if _head_component_state_hash(
            checked["head_state"], component=component
        ) != _head_component_state_hash(
            source_payloads[component]["head_state"], component=component
        ):
            raise ValueError(
                f"V27 final head {component} component state differs from source"
            )
    if _head_component_state_hash(
        source_payloads["quality"]["head_state"], component="knownness"
    ) != _head_component_state_hash(
        source_payloads["knownness"]["head_state"], component="knownness"
    ):
        raise ValueError("V27 quality stage changed the knownness source component")
    return checked, representation_manifest, base_record


def _calibration_sections(
    context: Any,
    *,
    task: str,
    score_payload: Mapping[str, Any],
) -> dict[str, Any]:
    cache = load_representation_cache(context, task=task, role="calibration")
    tensors = score_payload["tensors"]
    parent_ids = [str(value) for value in score_payload["parent_ids"]]
    cache_parent_ids = [str(value) for value in cache["parent_ids"]]
    score_labels = [str(value) for value in score_payload["label_names"]]
    cache_labels = [str(value) for value in cache["label_names"]]
    if parent_ids != cache_parent_ids or score_labels != cache_labels:
        raise ValueError("V27 calibration score/cache identity differs")
    lengths = tensors["lengths"].numpy().astype(np.int64)
    row = np.arange(len(lengths))
    final = lengths - 1
    labels = [str(value) for value in cache["label_names"]]
    logits = cache["tensors"]["sequence_logits"].numpy()[row, final]
    knownness_probability = 1.0 / (1.0 + np.exp(-tensors["knownness_logit"].numpy()[row, final]))
    quality_probability = 1.0 / (1.0 + np.exp(-tensors["quality_logit"].numpy()[row, final]))
    quality_target_np = _final_quality_targets(task, cache, context.config)
    known_target = _knownness_targets(task, labels).numpy().astype(np.int64)
    if task == V27_L1_TASK_NAME:
        class_mask = np.asarray([label in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"} for label in labels])
        class_labels = np.asarray([
            0 if label in {"DroneTarget", "BirdTarget", "BalloonTarget"} else 1
            for label in labels
        ], dtype=np.int64)
    else:
        class_mask = np.asarray([label in V27_L2_CLASS_NAMES for label in labels])
        class_labels = np.asarray([
            V27_L2_CLASS_NAMES.index(label) if label in V27_L2_CLASS_NAMES else -1
            for label in labels
        ], dtype=np.int64)
    class_logits = logits[class_mask]
    class_labels = class_labels[class_mask]
    temperature = _fit_temperature(
        class_logits,
        class_labels,
        lower=context.config.open_set.temperature_min,
        upper=context.config.open_set.temperature_max,
    )
    probabilities = _calibrated_probabilities(class_logits, temperature)
    known_scores = knownness_probability
    known_threshold, known_operating = _select_high_score_threshold(
        known_scores[known_target == 1],
        known_scores[known_target == 0],
        recall_floor=context.config.open_set.knownness_recall_floor,
    )
    quality_threshold, quality_operating = _select_high_score_threshold(
        quality_probability[quality_target_np == 1],
        quality_probability[quality_target_np == 0],
        recall_floor=context.config.open_set.quality_recall_floor,
    )
    section: dict[str, Any] = {
        "class_names": list(
            ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
            if task == V27_L1_TASK_NAME
            else V27_L2_CLASS_NAMES
        ),
        "temperature": float(temperature),
        "temperature_fit_support": int(class_labels.size),
        "temperature_fit_label_counts": {
            str(value): int(np.count_nonzero(class_labels == index))
            for index, value in enumerate(
                ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
                if task == V27_L1_TASK_NAME
                else V27_L2_CLASS_NAMES
            )
        },
        "knownness_threshold": float(known_threshold),
        "knownness_threshold_space": "sigmoid_probability",
        "knownness_score_direction": "higher_is_known",
        "knownness_positive_support": int(np.count_nonzero(known_target == 1)),
        "knownness_negative_support": int(np.count_nonzero(known_target == 0)),
        "knownness_operating_point": known_operating,
        "knownness_score_summary": _score_summary(known_target.tolist(), known_scores.tolist()),
        "conformal_error_rate": float(context.config.open_set.conformal_error_rate),
    }
    class_names = tuple(str(value) for value in section["class_names"])
    quality_negative_support = int(np.count_nonzero(quality_target_np == 0))
    required_quality_negative_support = int(
        context.config.open_set.quality_min_final_negative_support
    )
    quality_gate_active = (
        quality_negative_support >= required_quality_negative_support
    )
    section.update(
        _conformal_section(
            probabilities=probabilities,
            class_labels=class_labels,
            class_names=class_names,
            error_rate=context.config.open_set.conformal_error_rate,
        )
    )
    section["quality_threshold"] = float(quality_threshold)
    section["quality_threshold_space"] = "sigmoid_probability"
    section["quality_score_direction"] = "higher_is_eligible"
    section["quality_operating_point"] = quality_operating
    section["quality_positive_support"] = int(np.count_nonzero(quality_target_np == 1))
    section["quality_negative_support"] = quality_negative_support
    section["quality_gate"] = {
        "active": quality_gate_active,
        "status": (
            "active"
            if quality_gate_active
            else "inactive_insufficient_final_negative_support"
        ),
        "required_negative_support": required_quality_negative_support,
        "observed_negative_support": quality_negative_support,
        "learned_score_role": (
            "hard_pending_gate" if quality_gate_active else "advisory_only"
        ),
    }
    return {
        "section": section,
        "parent_ids": parent_ids,
        "labels": score_labels,
        "lengths": lengths,
        "logits": logits,
        "classifier_logits": logits,
        "knownness_probability": knownness_probability,
        "quality_probability": quality_probability,
        "deterministic_quality": quality_target_np.astype(bool),
        "known_target": known_target,
        "quality_target": quality_target_np,
        "class_mask": class_mask,
        "class_labels": class_labels,
        "class_probabilities": probabilities,
    }


def _conformal_section(
    *,
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    class_names: Sequence[str],
    error_rate: float,
) -> dict[str, Any]:
    """Fit one class-conditional conformal prediction-set contract.

    The returned fields are deliberately identical to the per-task registry
    section.  Calibration and replay therefore cannot drift by implementing
    candidate-set semantics in two places.
    """

    names = tuple(str(name) for name in class_names)
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(class_labels, dtype=np.int64)
    alpha = float(error_rate)
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or labels.shape != (values.shape[0],)
        or values.shape[0] <= 0
        or not np.isfinite(values).all()
        or not np.isfinite(labels).all()
        or not 0.0 < alpha < 1.0
        or set(labels.tolist()) - set(range(len(names)))
    ):
        raise ValueError("V27 conformal fitting inputs differ")
    nonconformity = 1.0 - values[np.arange(labels.size), labels]
    quantiles: dict[str, float] = {}
    floors: dict[str, float] = {}
    supports: dict[str, int] = {}
    coverages: dict[str, float] = {}
    for class_index, class_name in enumerate(names):
        class_nonconformity = nonconformity[labels == class_index]
        support = int(class_nonconformity.size)
        if support <= 0:
            raise ValueError(
                f"V27 calibration class has no conformal support: {class_name}"
            )
        rank = min(
            max(int(math.ceil((support + 1) * (1.0 - alpha))), 1),
            support,
        )
        quantile = float(np.sort(class_nonconformity)[rank - 1])
        quantiles[class_name] = quantile
        floors[class_name] = 1.0 - quantile
        supports[class_name] = support
        coverages[class_name] = float(
            np.mean(class_nonconformity <= quantile + 1.0e-12)
        )
    covered = np.asarray(
        [
            nonconformity[index]
            <= quantiles[names[int(labels[index])]] + 1.0e-12
            for index in range(int(labels.size))
        ],
        dtype=bool,
    )
    return {
        "conformal_error_rate": alpha,
        "conformal_method": (
            "class_conditional_split_conformal_1_minus_calibrated_probability_higher"
        ),
        "conformal_quantile_by_class": quantiles,
        "conformal_support_by_class": supports,
        "conformal_candidate_probability_floor_by_class": floors,
        "conformal_support": int(labels.size),
        "conformal_coverage_on_calibration": float(np.mean(covered)),
        "conformal_coverage_on_calibration_by_class": coverages,
    }


def _replace_conformal_section(
    section: Mapping[str, Any],
    *,
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    error_rate: float,
) -> dict[str, Any]:
    """Return ``section`` with one auditable candidate-set operating point."""

    value = dict(section)
    value.update(
        _conformal_section(
            probabilities=probabilities,
            class_labels=class_labels,
            class_names=tuple(str(name) for name in value["class_names"]),
            error_rate=error_rate,
        )
    )
    return value


def _knownness_threshold_grid(
    *,
    positive_scores: np.ndarray,
    baseline_threshold: float,
    grid_size: int,
) -> tuple[float, ...]:
    """Generate recall-safe knownness candidates without using raw labels twice.

    The one-dimensional score ranking was fitted by ``_select_high_score_threshold``.
    Joint action calibration may only relax it downwards, so every candidate
    preserves the per-head knownness recall floor before the cascade is tested.
    """

    positives = np.asarray(positive_scores, dtype=np.float64)
    if (
        positives.ndim != 1
        or positives.size <= 0
        or not np.isfinite(positives).all()
        or not math.isfinite(float(baseline_threshold))
        or not 0.0 <= float(baseline_threshold) <= 1.0
        or int(grid_size) <= 0
    ):
        raise ValueError("V27 knownness threshold grid inputs differ")
    eligible = np.unique(positives[positives <= float(baseline_threshold)])
    anchors = np.asarray([0.0, float(baseline_threshold)], dtype=np.float64)
    if eligible.size <= int(grid_size):
        selected = eligible
    else:
        positions = np.linspace(0, eligible.size - 1, num=int(grid_size), dtype=int)
        selected = eligible[positions]
    return tuple(
        float(value)
        for value in np.unique(np.concatenate((anchors, selected)))
    )


def _conformal_error_rate_grid(
    *,
    maximum_error_rate: float,
    grid_size: int,
) -> tuple[float, ...]:
    """Build one task-local conformal error-rate grid.

    L1 and L2 share the configured upper bound in the baseline, but their
    prediction-set supports and cascade roles differ.  Returning a fresh,
    validated grid per task makes that separation explicit and leaves room for
    future task-specific ranges without changing the action-selection schema.
    """

    maximum = float(maximum_error_rate)
    size = int(grid_size)
    if not math.isfinite(maximum) or not 0.0 < maximum < 1.0 or size <= 0:
        raise ValueError("V27 conformal error-rate grid inputs differ")
    values = np.linspace(maximum / size, maximum, num=size, dtype=np.float64)
    return tuple(float(value) for value in np.unique(values))


def _replay_l3_records(
    *,
    l1: Mapping[str, Any],
    l2: Mapping[str, Any],
    l1_section: Mapping[str, Any],
    l2_section: Mapping[str, Any],
    calibration_status: str,
    streaming: bool,
) -> list[dict[str, Any]]:
    """Resolve one aligned L1→L2 score population through the canonical L3.

    Both action calibration and development evaluation call this function.  It
    is intentionally the only score-to-verdict bridge in the V27 package.
    """

    if (
        l1["parent_ids"] != l2["parent_ids"]
        or l1["labels"] != l2["labels"]
        or len(l1["parent_ids"]) != len(l1["lengths"])
        or len(l2["parent_ids"]) != len(l2["lengths"])
    ):
        raise ValueError("V27 L1/L2 open-set role identity differs")
    l1_probabilities = _calibrated_probabilities(
        np.asarray(l1["classifier_logits"]), float(l1_section["temperature"])
    )
    l2_probabilities = _calibrated_probabilities(
        np.asarray(l2["classifier_logits"]), float(l2_section["temperature"])
    )
    records: list[dict[str, Any]] = []
    for index, parent_id in enumerate(l1["parent_ids"]):
        l1_candidates = _conformal_candidates(
            l1_probabilities[index],
            class_names=l1_section["class_names"],
            probability_floor_by_class=l1_section[
                "conformal_candidate_probability_floor_by_class"
            ],
        )
        l2_candidates = _conformal_candidates(
            l2_probabilities[index],
            class_names=l2_section["class_names"],
            probability_floor_by_class=l2_section[
                "conformal_candidate_probability_floor_by_class"
            ],
        )
        decision = resolve_l3_action(
            calibration_status=calibration_status,
            l1_target_accepted="KNOWN_TARGET_EVIDENCE" in l1_candidates,
            l1_clutter_accepted="KNOWN_CLUTTER_EVIDENCE" in l1_candidates,
            l1_knownness_accepted=bool(
                l1["knownness_probability"][index]
                >= float(l1_section["knownness_threshold"])
            ),
            l2_knownness_accepted=bool(
                l2["knownness_probability"][index]
                >= float(l2_section["knownness_threshold"])
            ),
            dbb_conformal_candidates=l2_candidates,
            streaming=streaming,
            l1_deterministic_quality_eligible=bool(
                l1["deterministic_quality"][index]
            ),
            l1_learned_quality_eligible=(
                bool(
                    l1["quality_probability"][index]
                    >= float(l1_section["quality_threshold"])
                )
                if bool(l1_section["quality_gate"]["active"])
                else None
            ),
            l1_learned_quality_gate_active=bool(
                l1_section["quality_gate"]["active"]
            ),
            l2_deterministic_quality_eligible=bool(
                l2["deterministic_quality"][index]
            ),
            l2_learned_quality_eligible=(
                bool(
                    l2["quality_probability"][index]
                    >= float(l2_section["quality_threshold"])
                )
                if bool(l2_section["quality_gate"]["active"])
                else None
            ),
            l2_learned_quality_gate_active=bool(
                l2_section["quality_gate"]["active"]
            ),
        )
        records.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": str(l1["labels"][index]),
                "event_count": int(l1["lengths"][index]),
                "action": str(decision.action),
                "reason_code": decision.reason_code,
                "l1_target_probability": float(l1_probabilities[index, 0]),
                "l1_clutter_probability": float(l1_probabilities[index, 1]),
                "l1_conformal_candidates": list(l1_candidates),
                "l1_knownness_probability": float(
                    l1["knownness_probability"][index]
                ),
                "l1_quality_probability": float(l1["quality_probability"][index]),
                "l1_deterministic_quality_eligible": bool(
                    l1["deterministic_quality"][index]
                ),
                "l2_knownness_probability": float(
                    l2["knownness_probability"][index]
                ),
                "l2_quality_probability": float(l2["quality_probability"][index]),
                "l2_deterministic_quality_eligible": bool(
                    l2["deterministic_quality"][index]
                ),
                "l1_learned_quality_gate_active": bool(
                    l1_section["quality_gate"]["active"]
                ),
                "l2_learned_quality_gate_active": bool(
                    l2_section["quality_gate"]["active"]
                ),
                "l2_probabilities": {
                    name: float(probability)
                    for name, probability in zip(
                        V27_L2_CLASS_NAMES,
                        l2_probabilities[index],
                        strict=True,
                    )
                },
                "dbb_conformal_candidates": list(l2_candidates),
            }
        )
    return records


def _action_safety_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate cascade safety endpoints from OOD diagnostic performance."""

    all_known_target = [
        record
        for record in records
        if str(record["label_name"]) in V27_L2_CLASS_NAMES
    ]
    known_target = [
        record
        for record in all_known_target
        if bool(record["l1_deterministic_quality_eligible"])
        and bool(record["l2_deterministic_quality_eligible"])
    ]
    all_clutter = [
        record
        for record in records
        if str(record["label_name"]) == "ClutterTarget"
    ]
    clutter = [
        record
        for record in all_clutter
        if bool(record["l1_deterministic_quality_eligible"])
    ]
    unknown = [
        record for record in records if str(record["label_name"]) == "UnknownTarget"
    ]
    if not known_target or not clutter or not unknown:
        raise ValueError("V27 action calibration lacks required mixed support")
    dbb_route_accepted = sum(
        str(record["action"]) in V27_L2_CLASS_NAMES for record in known_target
    )
    dbb_exact = sum(
        str(record["action"]) == str(record["label_name"])
        for record in known_target
    )
    clutter_accepted = sum(
        str(record["action"]) == "ClutterTarget" for record in clutter
    )
    known_pending_or_rejected = sum(
        str(record["action"]) in {"PENDING_ABSTAIN", "UNKNOWN_REJECTED"}
        for record in (*known_target, *clutter)
    )
    unknown_ood_rejected = sum(
        str(record["action"]) == "UNKNOWN_REJECTED"
        and str(record["reason_code"]) == "OOD_NONCONFORMING"
        for record in unknown
    )
    unknown_low_quality_pending = sum(
        str(record["reason_code"]) == "LOW_QUALITY_OR_IMMATURE"
        for record in unknown
    )
    known_target_non_route = [
        record
        for record in known_target
        if str(record["action"]) not in V27_L2_CLASS_NAMES
    ]
    clutter_nonaccepted = [
        record
        for record in clutter
        if str(record["action"]) != "ClutterTarget"
    ]
    return {
        "all_known_target_parent_count": len(all_known_target),
        "eligible_known_target_support": len(known_target),
        "known_target_deterministic_quality_pending_count": (
            len(all_known_target) - len(known_target)
        ),
        "eligible_known_target_dbb_route_accepted_count": dbb_route_accepted,
        "eligible_known_target_dbb_route_recall": dbb_route_accepted
        / len(known_target),
        "eligible_known_target_non_dbb_route_count": len(known_target_non_route),
        "eligible_known_target_non_dbb_route_action_counts": dict(
            sorted(Counter(str(record["action"]) for record in known_target_non_route).items())
        ),
        "eligible_known_target_non_dbb_route_reason_counts": dict(
            sorted(
                Counter(str(record["reason_code"]) for record in known_target_non_route).items()
            )
        ),
        "eligible_known_target_exact_dbb_class_correct_count": dbb_exact,
        "eligible_known_target_exact_dbb_class_correct_rate": dbb_exact
        / len(known_target),
        "all_clutter_parent_count": len(all_clutter),
        "eligible_clutter_support": len(clutter),
        "clutter_deterministic_quality_pending_count": len(all_clutter)
        - len(clutter),
        "eligible_clutter_accepted_count": clutter_accepted,
        "eligible_clutter_recall": clutter_accepted / len(clutter),
        "eligible_clutter_nonaccepted_count": len(clutter_nonaccepted),
        "eligible_clutter_nonaccepted_action_counts": dict(
            sorted(Counter(str(record["action"]) for record in clutter_nonaccepted).items())
        ),
        "eligible_clutter_nonaccepted_reason_counts": dict(
            sorted(
                Counter(str(record["reason_code"]) for record in clutter_nonaccepted).items()
            )
        ),
        "all_known_parent_count": len(all_known_target) + len(all_clutter),
        "eligible_known_parent_count": len(known_target) + len(clutter),
        "eligible_known_pending_or_unknown_rejected_count": known_pending_or_rejected,
        "unknown_parent_count": len(unknown),
        "unknown_ood_nonconforming_rejected_count": unknown_ood_rejected,
        "unknown_ood_nonconforming_rejection_rate": unknown_ood_rejected
        / len(unknown),
        "unknown_low_quality_pending_count": unknown_low_quality_pending,
        "low_quality_pending_counted_as_unknown_rejection": False,
    }


def _action_safety_floor_assessment(
    metrics: Mapping[str, Any],
    *,
    known_target_floor: float,
    clutter_floor: float,
) -> dict[str, Any]:
    """Assess registered action floors without changing a fitted operating point."""

    known_floor = float(known_target_floor)
    clutter_floor_value = float(clutter_floor)
    if (
        not math.isfinite(known_floor)
        or not math.isfinite(clutter_floor_value)
        or not 0.0 < known_floor <= 1.0
        or not 0.0 < clutter_floor_value <= 1.0
    ):
        raise ValueError("V27 action safety floors differ")
    known_recall = float(metrics["eligible_known_target_dbb_route_recall"])
    clutter_recall = float(metrics["eligible_clutter_recall"])
    if (
        not math.isfinite(known_recall)
        or not math.isfinite(clutter_recall)
        or not 0.0 <= known_recall <= 1.0
        or not 0.0 <= clutter_recall <= 1.0
    ):
        raise ValueError("V27 action safety metrics differ")
    known_met = known_recall + 1.0e-12 >= known_floor
    clutter_met = clutter_recall + 1.0e-12 >= clutter_floor_value
    return {
        "known_target_action_floor": known_floor,
        "eligible_known_target_dbb_route_recall": known_recall,
        "known_target_action_floor_met": known_met,
        "clutter_action_floor": clutter_floor_value,
        "eligible_clutter_recall": clutter_recall,
        "clutter_action_floor_met": clutter_met,
        "status": "passed" if known_met and clutter_met else "failed",
    }


def _joint_action_calibration(
    context: Any,
    *,
    l1: Mapping[str, Any],
    l2: Mapping[str, Any],
    l1_section: Mapping[str, Any],
    l2_section: Mapping[str, Any],
    l1_class_probabilities: np.ndarray,
    l2_class_probabilities: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select L3 parameters only after replaying the full cascade.

    Candidate thresholds are limited to individually recall-safe knownness
    values and a registered conformal-error grid.  The ranking first satisfies
    the known-target and Clutter action floors, then maximises true OOD
    rejection, exact DBB correctness and route recall.  If no action-safe
    point exists, the caller receives a diagnostic but no fitted registry.
    """

    config = context.config.open_set
    l1_known_target = np.asarray(l1["known_target"], dtype=np.int64)
    l2_known_target = np.asarray(l2["known_target"], dtype=np.int64)
    if l1_known_target.shape != np.asarray(l1["knownness_probability"]).shape or (
        l2_known_target.shape != np.asarray(l2["knownness_probability"]).shape
    ):
        raise ValueError("V27 joint action calibration knownness target shape differs")
    l1_candidates = _knownness_threshold_grid(
        positive_scores=np.asarray(l1["knownness_probability"])[l1_known_target == 1],
        baseline_threshold=float(l1_section["knownness_threshold"]),
        grid_size=config.action_knownness_threshold_grid_size,
    )
    l2_candidates = _knownness_threshold_grid(
        positive_scores=np.asarray(l2["knownness_probability"])[l2_known_target == 1],
        baseline_threshold=float(l2_section["knownness_threshold"]),
        grid_size=config.action_knownness_threshold_grid_size,
    )
    l1_error_rates = _conformal_error_rate_grid(
        maximum_error_rate=config.conformal_error_rate,
        grid_size=config.action_conformal_error_rate_grid_size,
    )
    l2_error_rates = _conformal_error_rate_grid(
        maximum_error_rate=config.conformal_error_rate,
        grid_size=config.action_conformal_error_rate_grid_size,
    )
    candidates_considered = 0
    feasible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any], dict[str, Any]]]
    feasible = []
    for l1_error_rate in l1_error_rates:
        l1_conformal = _replace_conformal_section(
            l1_section,
            probabilities=l1_class_probabilities,
            class_labels=np.asarray(l1["class_labels"], dtype=np.int64),
            error_rate=l1_error_rate,
        )
        for l2_error_rate in l2_error_rates:
            l2_conformal = _replace_conformal_section(
                l2_section,
                probabilities=l2_class_probabilities,
                class_labels=np.asarray(l2["class_labels"], dtype=np.int64),
                error_rate=l2_error_rate,
            )
            for l1_threshold in l1_candidates:
                for l2_threshold in l2_candidates:
                    candidates_considered += 1
                    l1_trial = {
                        **l1_conformal,
                        "knownness_threshold": float(l1_threshold),
                    }
                    l2_trial = {
                        **l2_conformal,
                        "knownness_threshold": float(l2_threshold),
                    }
                    records = _replay_l3_records(
                        l1=l1,
                        l2=l2,
                        l1_section=l1_trial,
                        l2_section=l2_trial,
                        calibration_status="fitted",
                        streaming=False,
                    )
                    metrics = _action_safety_metrics(records)
                    action_safety = _action_safety_floor_assessment(
                        metrics,
                        known_target_floor=config.action_known_target_recall_floor,
                        clutter_floor=config.action_clutter_recall_floor,
                    )
                    if action_safety["status"] == "passed":
                        ranking = (
                            metrics["unknown_ood_nonconforming_rejection_rate"],
                            metrics["eligible_known_target_exact_dbb_class_correct_rate"],
                            metrics["eligible_known_target_dbb_route_recall"],
                            metrics["eligible_clutter_recall"],
                            l1_error_rate,
                            l2_error_rate,
                            float(l1_threshold),
                            float(l2_threshold),
                        )
                        feasible.append((ranking, l1_trial, l2_trial, metrics))
    selection_contract = {
        "method": ACTION_SELECTION_METHOD,
        "known_target_action_floor": config.action_known_target_recall_floor,
        "clutter_action_floor": config.action_clutter_recall_floor,
        "knownness_candidate_policy": "per_head_recall_safe_thresholds_at_or_below_independent_operating_point",
        "l1_conformal_error_rate_grid": list(l1_error_rates),
        "l2_conformal_error_rate_grid": list(l2_error_rates),
        "selection_ranking": [
            "unknown_ood_nonconforming_rejection_rate",
            "eligible_known_target_exact_dbb_class_correct_rate",
            "eligible_known_target_dbb_route_recall",
            "eligible_clutter_recall",
            "l1_conformal_error_rate",
            "l2_conformal_error_rate",
            "l1_knownness_threshold",
            "l2_knownness_threshold",
        ],
        "candidates_considered": candidates_considered,
        "feasible_candidate_count": len(feasible),
        "low_quality_pending_counted_as_unknown_rejection": False,
    }
    if not feasible:
        raise ValueError(
            "V27 joint action calibration found no point satisfying known-target "
            "and Clutter safety floors"
        )
    _, selected_l1, selected_l2, metrics = max(feasible, key=lambda item: item[0])
    selection_contract["selected"] = {
        "l1_conformal_error_rate": selected_l1["conformal_error_rate"],
        "l2_conformal_error_rate": selected_l2["conformal_error_rate"],
        "l1_knownness_threshold": selected_l1["knownness_threshold"],
        "l2_knownness_threshold": selected_l2["knownness_threshold"],
        "action_metrics": metrics,
    }
    return selected_l1, selected_l2, selection_contract


def _validate_l3_registry_artifacts(context: Any, registry: Mapping[str, Any]) -> None:
    """Verify artifacts referenced by the current L3 registry.

    An ``unfitted`` registry is a valid fail-closed state and intentionally has
    no head checkpoints or score caches to dereference.  Only a fitted registry
    carries the downstream artifacts checked below.
    """

    checked = verify_l3_registry(
        registry,
        expected_system_manifest_sha256=context.manifest_sha256,
    )
    if checked.get("status") == "unfitted":
        return
    scores = checked.get("fitted_scores")
    if not isinstance(scores, Mapping):
        raise ValueError("V27 fitted L3 registry lacks score payload")
    representation_manifest = load_representation_cache_manifest(context)
    pool_registry, _ = load_open_set_pool_artifacts(context)
    if (
        scores.get("representation_cache_manifest_sha256")
        != representation_manifest["representation_cache_manifest_sha256"]
        or scores.get("calibration_pool_registry_sha256")
        != pool_registry["registry_sha256"]
    ):
        raise ValueError("V27 fitted L3 registry artifact binding differs")
    head_checkpoints = scores.get("head_checkpoints")
    score_caches = scores.get("calibration_score_cache_sha256")
    if not isinstance(head_checkpoints, Mapping) or not isinstance(score_caches, Mapping):
        raise ValueError("V27 fitted L3 registry provenance is incomplete")
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        head_entry = head_checkpoints.get(task)
        if not isinstance(head_entry, Mapping):
            raise ValueError(f"V27 fitted L3 registry head entry is missing: {task}")
        head_path = resolve_project_path(
            context.project_root, str(head_entry.get("path", "")), must_exist=True
        )
        if sha256_file(head_path) != head_entry.get("sha256"):
            raise ValueError(f"V27 fitted L3 head file hash differs: {task}")
        head_payload, _, _ = _load_final_head(
            context, task=task, seed=int(scores.get("seed", -1))
        )
        if (
            head_payload.get("trajectory_head_state_sha256")
            != head_entry.get("trajectory_head_state_sha256")
        ):
            raise ValueError(f"V27 fitted L3 head state binding differs: {task}")
        score_cache = load_head_score_cache(
            context,
            task=task,
            role="calibration",
            seed=int(scores.get("seed", -1)),
        )
        if score_cache.get("score_cache_sha256") != score_caches.get(task):
            raise ValueError(f"V27 fitted L3 score cache hash differs: {task}")
        if (
            score_cache.get("head_state_sha256")
            != head_payload.get("trajectory_head_state_sha256")
        ):
            raise ValueError(
                f"V27 fitted L3 score cache head-state binding differs: {task}"
            )
    action_selection = scores.get("action_selection")
    if not isinstance(action_selection, Mapping):
        raise ValueError("V27 fitted L3 action-selection record is missing")
    selected = action_selection.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("action_metrics"), Mapping
    ):
        raise ValueError("V27 fitted L3 action-selection result is incomplete")
    replay_metrics = _action_safety_metrics(
        _replay_l3_records(
            l1=_final_score_arrays(
                context,
                task=V27_L1_TASK_NAME,
                role="calibration",
                seed=int(scores.get("seed", -1)),
            ),
            l2=_final_score_arrays(
                context,
                task=V27_L2_TASK_NAME,
                role="calibration",
                seed=int(scores.get("seed", -1)),
            ),
            l1_section=scores["l1"],
            l2_section=scores["l2"],
            calibration_status="fitted",
            streaming=False,
        )
    )
    if dict(selected["action_metrics"]) != replay_metrics:
        raise ValueError("V27 fitted L3 action-selection replay differs")
    if (
        replay_metrics["eligible_known_target_dbb_route_recall"] + 1.0e-12
        < context.config.open_set.action_known_target_recall_floor
        or replay_metrics["eligible_clutter_recall"] + 1.0e-12
        < context.config.open_set.action_clutter_recall_floor
    ):
        raise ValueError("V27 fitted L3 action-selection no longer meets safety floors")


def _final_score_arrays(
    context: Any,
    *,
    task: str,
    role: str,
    seed: int,
) -> dict[str, Any]:
    """Return one verified role at final causal state resolution."""

    payload = load_head_score_cache(context, task=task, role=role, seed=seed)
    head_payload, _, _ = _load_final_head(context, task=task, seed=seed)
    if payload.get("head_state_sha256") != head_payload.get(
        "trajectory_head_state_sha256"
    ):
        raise ValueError("V27 replay score cache/final head binding differs")
    tensors = payload["tensors"]
    lengths = tensors["lengths"].detach().cpu().numpy().astype(np.int64)
    rows = np.arange(lengths.size)
    final = lengths - 1
    knownness_logit = tensors["knownness_logit"].detach().cpu().numpy()[rows, final]
    quality_logit = tensors["quality_logit"].detach().cpu().numpy()[rows, final]
    logits = tensors["sequence_logits"].detach().cpu().numpy()[rows, final]
    return {
        "parent_ids": [str(value) for value in payload["parent_ids"]],
        "labels": [str(value) for value in payload["label_names"]],
        "lengths": lengths,
        "knownness_probability": 1.0 / (1.0 + np.exp(-knownness_logit)),
        "quality_probability": 1.0 / (1.0 + np.exp(-quality_logit)),
        "classifier_logits": logits,
        "deterministic_quality": _final_quality_targets(
            task, payload, context.config
        ).astype(bool),
        "score_cache_sha256": payload["score_cache_sha256"],
    }


def _conformal_candidates(
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    probability_floor_by_class: Mapping[str, Any],
) -> tuple[str, ...]:
    names = tuple(str(name) for name in class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("V27 conformal class names are invalid")
    if probabilities.shape != (len(names),):
        raise ValueError("V27 calibrated probability shape differs")
    if not isinstance(probability_floor_by_class, Mapping) or set(
        probability_floor_by_class
    ) != set(names):
        raise ValueError("V27 class-conditional conformal floors differ")
    return tuple(
        name
        for name, probability in zip(names, probabilities, strict=True)
        if float(probability) >= float(probability_floor_by_class[name])
    )


def _decision_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record["action"]) for record in records)
    reasons = Counter(str(record["reason_code"]) for record in records)
    by_label: dict[str, dict[str, int]] = {}
    for record in records:
        label = str(record["label_name"])
        entry = by_label.setdefault(label, {})
        action = str(record["action"])
        entry[action] = entry.get(action, 0) + 1
    return {
        "support": len(records),
        "action_counts": dict(sorted(counts.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "action_counts_by_label": {
            label: dict(sorted(values.items()))
            for label, values in sorted(by_label.items())
        },
    }


def evaluate_open_set_v27(
    context: Any,
    *,
    role: str = "representation_development",
    seed: int | None = None,
) -> dict[str, Any]:
    """Replay the calibrated L1→L2→L3 cascade on a non-test role.

    The result distinguishes Pending, OOD rejection, Clutter, and DBB class
    ambiguity.  It is an explicitly labelled development/calibration replay,
    never a deployment claim or a test-set evaluation.
    """

    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    role = _require_role(role)
    if role != "representation_development":
        raise ValueError(
            "V27 open-set replay is reserved for representation_development; "
            "calibration is fitting-only and representation_fit is training-only"
        )
    registry_path = resolve_project_path(
        context.project_root,
        context.config.l3.calibration_registry_path,
        must_exist=True,
    )
    registry = read_v27_sealed_json(registry_path)
    _validate_l3_registry_artifacts(context, registry)
    scores = registry["fitted_scores"]
    l1_section = scores["l1"]
    l2_section = scores["l2"]
    l1 = _final_score_arrays(
        context, task=V27_L1_TASK_NAME, role=role, seed=actual_seed
    )
    l2 = _final_score_arrays(
        context, task=V27_L2_TASK_NAME, role=role, seed=actual_seed
    )
    records = _replay_l3_records(
        l1=l1,
        l2=l2,
        l1_section=l1_section,
        l2_section=l2_section,
        calibration_status="fitted",
        streaming=False,
    )
    summary = _decision_summary(records)
    unknown_records = [
        record for record in records if record["label_name"] == "UnknownTarget"
    ]
    low_quality_unknown = [
        record
        for record in unknown_records
        if record["reason_code"] == "LOW_QUALITY_OR_IMMATURE"
    ]
    true_ood_rejections = [
        record
        for record in unknown_records
        if record["action"] == "UNKNOWN_REJECTED"
        and record["reason_code"] == "OOD_NONCONFORMING"
    ]
    action_safety = _action_safety_metrics(records)
    action_safety_floor_assessment = _action_safety_floor_assessment(
        action_safety,
        known_target_floor=context.config.open_set.action_known_target_recall_floor,
        clutter_floor=context.config.open_set.action_clutter_recall_floor,
    )
    report = {
        "status": "completed_open_set_development_replay",
        "schema_version": "RADAR-LN3-V27-open-set-evaluation-v1",
        "experiment_id": "V27",
        "role": role,
        "seed": actual_seed,
        "system_manifest_sha256": context.manifest_sha256,
        "l3_registry_sha256": registry["registry_sha256"],
        "representation_cache_manifest_sha256": scores[
            "representation_cache_manifest_sha256"
        ],
        "calibration_pool_registry_sha256": scores[
            "calibration_pool_registry_sha256"
        ],
        "score_cache_sha256": {
            V27_L1_TASK_NAME: l1["score_cache_sha256"],
            V27_L2_TASK_NAME: l2["score_cache_sha256"],
        },
        "decision_summary": summary,
        "safety": {
            **action_safety,
            "floor_assessment": action_safety_floor_assessment,
            # Retain the historical summary names for consumers that have not
            # yet migrated, while the eligible action metrics above remain the
            # authoritative product-safety endpoints.
            "known_parent_count": len(
                [record for record in records if record["label_name"] != "UnknownTarget"]
            ),
            "unknown_parent_count": len(unknown_records),
            "unknown_ood_nonconforming_rejected_count": len(true_ood_rejections),
            "unknown_low_quality_pending_count": len(low_quality_unknown),
            "l1_route_policy": "dynamic_temperature_split_conformal_probability_set",
        },
        "records": records,
        "unknown_classifier_training_exposure": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
    }
    report_path = resolve_project_path(
        context.project_root, context.config.artifacts.open_set_evaluation_report_path
    )
    _write_derived_sealed_json(context, report_path, report)
    predictions_path = resolve_project_path(
        context.project_root, context.config.artifacts.open_set_predictions_path
    )
    _write_derived_sealed_json(
        context,
        predictions_path,
        {
            "status": "completed_open_set_predictions",
            "schema_version": "RADAR-LN3-V27-open-set-predictions-v1",
            "experiment_id": "V27",
            "role": role,
            "seed": actual_seed,
            "system_manifest_sha256": context.manifest_sha256,
            "l3_registry_sha256": registry["registry_sha256"],
            "records": records,
            "test_payload_access_count": 0,
        },
    )
    return {**report, "report_path": str(report_path), "predictions_path": str(predictions_path)}


def audit_open_set_v27(context: Any, *, seed: int | None = None) -> dict[str, Any]:
    """Audit V27 open-set provenance without opening test data.

    An unfitted L3 registry is a valid, deliberately fail-closed intermediate
    state.  At that point representation and pool provenance can still be
    audited, but final heads and score caches do not exist yet.  Treating that
    state as a complete audit used to produce a misleading ``FileNotFoundError``
    rather than an explicit readiness result.
    """

    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    representation_manifest = load_representation_cache_manifest(context)
    pool_registry, _ = load_open_set_pool_artifacts(context)
    registry_path = resolve_project_path(
        context.project_root,
        context.config.l3.calibration_registry_path,
        must_exist=True,
    )
    registry = read_v27_sealed_json(registry_path)
    _validate_l3_registry_artifacts(context, registry)
    frozen_binding = load_frozen_base_model(
        context, task=V27_L1_TASK_NAME, device=torch.device("cpu")
    )[2]
    common = {
        "schema_version": "RADAR-LN3-V27-open-set-audit-v1",
        "experiment_id": "V27",
        "seed": actual_seed,
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": frozen_binding["binding_sha256"],
        "representation_cache_manifest_sha256": representation_manifest[
            "representation_cache_manifest_sha256"
        ],
        "open_set_pool_registry_sha256": pool_registry["registry_sha256"],
        "l3_registry_sha256": registry["registry_sha256"],
        "unknown_is_model_class": False,
        "unknown_classifier_training_exposure": 0,
        "test_payload_access_count": 0,
        "fixed_softmax_threshold_allowed": False,
        "single_distance_gate_allowed": False,
        "deployment_eligible": False,
    }
    if registry.get("status") == "unfitted":
        audit = {
            "status": "not_ready_pending_head_training",
            **common,
            "tasks": {},
            "readiness": {
                "l3_registry_status": "unfitted_fail_closed",
                "final_heads_present": False,
                "calibration_score_caches_present": False,
                "next_required_stages": [
                    "train-knownness",
                    "train-quality",
                    "finalize-heads",
                    "calibrate",
                ],
            },
        }
        audit_path = resolve_project_path(
            context.project_root, context.config.artifacts.open_set_audit_report_path
        )
        _write_derived_sealed_json(context, audit_path, audit)
        return {**audit, "report_path": str(audit_path)}

    task_checks: dict[str, Any] = {}
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        head, _, base_record = _load_final_head(context, task=task, seed=actual_seed)
        task_checks[task] = {
            "frozen_base_checkpoint_sha256": base_record["checkpoint_sha256"],
            "frozen_classifier_state_sha256": base_record[
                "classifier_state_sha256"
            ],
            "final_head_trajectory_state_sha256": head[
                "trajectory_head_state_sha256"
            ],
            "final_head_component_provenance": head["component_provenance"],
            "calibration_score_cache_sha256": load_head_score_cache(
                context,
                task=task,
                role="calibration",
                seed=actual_seed,
            )["score_cache_sha256"],
        }
    audit = {
        "status": "passed_open_set_audit",
        **common,
        "tasks": task_checks,
        "readiness": {
            "l3_registry_status": "fitted",
            "final_heads_present": True,
            "calibration_score_caches_present": True,
        },
    }
    audit_path = resolve_project_path(
        context.project_root, context.config.artifacts.open_set_audit_report_path
    )
    _write_derived_sealed_json(context, audit_path, audit)
    return {**audit, "report_path": str(audit_path)}


def calibrate_v27(context: Any, *, seed: int | None = None) -> dict[str, Any]:
    """Fit dynamic score calibration and an action-safe L1→L2→L3 registry."""

    actual_seed = int(context.config.training.seeds[0] if seed is None else seed)
    if actual_seed not in context.config.training.seeds:
        raise ValueError("V27 calibration seed is not registered")
    representation_manifest = load_representation_cache_manifest(context)
    pool_registry, _ = load_open_set_pool_artifacts(context)
    section_results: dict[str, Any] = {}
    score_caches: dict[str, Any] = {}
    for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME):
        head_payload, _, base_record = _load_final_head(
            context, task=task, seed=actual_seed
        )
        score_payload = _with_head_scores(
            context,
            task=task,
            role="calibration",
            head_payload=head_payload,
        )
        score_payload = _write_head_score_cache(
            context, task=task, role="calibration", payload=score_payload
        )
        score_caches[task] = score_payload
        section_results["l1" if task == V27_L1_TASK_NAME else "l2"] = (
            _calibration_sections(
                context, task=task, score_payload=score_payload
            )
        )
    selected_l1, selected_l2, action_selection = _joint_action_calibration(
        context,
        l1=section_results["l1"],
        l2=section_results["l2"],
        l1_section=section_results["l1"]["section"],
        l2_section=section_results["l2"]["section"],
        l1_class_probabilities=section_results["l1"]["class_probabilities"],
        l2_class_probabilities=section_results["l2"]["class_probabilities"],
    )
    sections = {"l1": selected_l1, "l2": selected_l2}
    score_contract = {
        "schema_version": CALIBRATION_SCHEMA,
        "knownness_score": "sigmoid(trajectory_head_logit); higher_is_known",
        "quality_score": "sigmoid(quality_head_logit); higher_is_eligible",
        "classifier_probability": "softmax(logits / dynamic_temperature)",
        "classifier_probability_set": "dynamic_temperature_class_conditional_conformal_probability_set",
        "conformal_nonconformity": "1 - calibrated_probability_of_candidate",
        "fixed_softmax_threshold": False,
        "single_distance_gate": False,
        "quality_label_policy": "causal_deterministic_facts_v1",
    }
    calibration_payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA,
        "experiment_id": "V27",
        "status": "fitted",
        "fitting_method": (
            "independent_calibration_pool_dynamic_temperature_per_task_conformal_"
            "joint_l1_l2_l3_action_replay_v3"
        ),
        "seed": actual_seed,
        "system_manifest_sha256": context.manifest_sha256,
        "frozen_base_binding_sha256": representation_manifest[
            "frozen_base_binding_sha256"
        ],
        "representation_cache_manifest_sha256": representation_manifest[
            "representation_cache_manifest_sha256"
        ],
        "calibration_pool_registry_sha256": pool_registry["registry_sha256"],
        "calibration_pool_parent_count": int(
            pool_registry["roles"]["calibration"]["parent_count"]
        ),
        "calibration_scope": {
            "pool_role": "calibration",
            "head_optimization_independent": True,
            # The frozen classification endpoints were selected on validation,
            # and the action grid selects alpha/thresholds on this same pool.
            # This is therefore independent of head fitting but not a
            # deployment-grade post-selection conformal guarantee.
            "classifier_checkpoint_selection_independent": False,
            "coverage_guarantee": (
                "not_claimed_after_joint_action_selection_or_classifier_"
                "checkpoint_selection"
            ),
        },
        "score_contract": score_contract,
        "score_contract_sha256": canonical_sha256(score_contract),
        "action_selection": action_selection,
        "l1": sections["l1"],
        "l2": sections["l2"],
        "quality": {
            "threshold": float(
                max(sections["l1"]["quality_threshold"], sections["l2"]["quality_threshold"])
            ),
            "threshold_space": "sigmoid_probability",
            "score_direction": "higher_is_eligible",
            "positive_support": int(
                min(
                    sections["l1"]["quality_positive_support"],
                    sections["l2"]["quality_positive_support"],
                )
            ),
            "negative_support": int(
                min(
                    sections["l1"]["quality_negative_support"],
                    sections["l2"]["quality_negative_support"],
                )
            ),
            "task_specific_thresholds": {
                "known_gate": sections["l1"]["quality_threshold"],
                "dbb_specialist": sections["l2"]["quality_threshold"],
            },
            "task_specific_gate_active": {
                "known_gate": bool(sections["l1"]["quality_gate"]["active"]),
                "dbb_specialist": bool(sections["l2"]["quality_gate"]["active"]),
            },
        },
        "head_checkpoints": {
            task: {
                "path": relative_project_path(
                    context.project_root,
                    _head_stage_path(context, task=task, stage="final", seed=actual_seed),
                ),
                "sha256": sha256_file(
                    _head_stage_path(context, task=task, stage="final", seed=actual_seed)
                ),
                "trajectory_head_state_sha256": _load_final_head(
                    context, task=task, seed=actual_seed
                )[0]["trajectory_head_state_sha256"],
            }
            for task in (V27_L1_TASK_NAME, V27_L2_TASK_NAME)
        },
        "calibration_score_cache_sha256": {
            task: score_caches[task]["score_cache_sha256"]
            for task in score_caches
        },
        "unknown_used_for_classifier_training": False,
        "test_payload_access_count": 0,
    }
    registry = make_fitted_registry(
        system_manifest_sha256=context.manifest_sha256,
        fitted_scores=calibration_payload,
    )
    registry_path = resolve_project_path(
        context.project_root, context.config.l3.calibration_registry_path
    )
    if registry_path.exists():
        existing = read_v27_sealed_json(registry_path)
        status = existing.get("status")
        if status == "fitted":
            checked_existing = verify_l3_registry(
                existing,
                expected_system_manifest_sha256=context.manifest_sha256,
            )
            if checked_existing["registry_sha256"] != registry["registry_sha256"]:
                raise FileExistsError(
                    "refusing to replace fitted V27 L3 calibration evidence; "
                    "prepare a new manifest-bound registry path for a new fit"
                )
            write_v27_sealed_json_if_absent_or_identical(registry_path, registry)
        elif status == "unfitted":
            verify_l3_registry(
                existing,
                expected_system_manifest_sha256=context.manifest_sha256,
            )
            # The unfitted registry is an explicit fail-closed placeholder,
            # rather than evidence.  Replacing it with the just-fitted payload
            # is the only permitted mutable transition at this path.
            write_v27_sealed_json(registry_path, registry)
        else:
            raise ValueError("V27 L3 registry status is invalid")
    else:
        write_v27_sealed_json_if_absent_or_identical(registry_path, registry)
    checked = verify_l3_registry(
        read_v27_sealed_json(registry_path),
        expected_system_manifest_sha256=context.manifest_sha256,
    )
    return {
        "status": "fitted",
        "registry_path": str(registry_path),
        "registry_sha256": checked["registry_sha256"],
        "calibration_pool_registry_sha256": pool_registry["registry_sha256"],
        "action_selection": action_selection,
        "sections": {name: value for name, value in sections.items()},
        "test_payload_access_count": 0,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "HEAD_CHECKPOINT_SCHEMA",
    "HEAD_SUMMARY_SCHEMA",
    "LABEL_TO_CODE",
    "OPEN_SET_ROLES",
    "REPRESENTATION_CACHE_SCHEMA",
    "REPRESENTATION_MANIFEST_SCHEMA",
    "calibrate_v27",
    "audit_open_set_v27",
    "deterministic_quality_targets",
    "evaluate_open_set_v27",
    "finalize_open_set_heads",
    "load_head_score_cache",
    "load_representation_cache",
    "load_representation_cache_manifest",
    "materialize_head_score_cache",
    "materialize_representation_caches",
    "train_open_set_heads",
]
