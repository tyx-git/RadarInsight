"""Causal V27d representation variants and quality population masks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from .endpoint import ENDPOINT_CACHE_SCHEMA, _endpoint_manifest_entry, endpoint_cache_digest, load_endpoint_cache, load_endpoint_manifest
from .pools import load_pools
from .protocol import V27DContext, V27D_KNOWN_LABELS, V27D_ROLES, V27D_VARIANTS, artifact_path, project_path, read_sealed_json


FEATURE_CACHE_SCHEMA = "RADAR-LN3-V27d-causal-knownness-feature-cache-v2"
FEATURE_MANIFEST_SCHEMA = "RADAR-LN3-V27d-feature-cache-manifest-v1"
def _event_trajectory_dim(context: V27DContext) -> int:
    return context.config.data.liquid_hidden_dim * 7 + context.config.data.event_fact_dim


def _summary_feature_dim(context: V27DContext) -> int:
    return context.config.data.liquid_hidden_dim * 14


def _raw_slices(hidden_dim: int) -> dict[str, slice]:
    return {
        "raw": slice(0, hidden_dim),
        "candidate": slice(hidden_dim, hidden_dim * 2),
        "mixed": slice(hidden_dim * 2, hidden_dim * 3),
        "update": slice(hidden_dim * 3, hidden_dim * 4),
    }


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d feature tensor is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def _padded_values_are_zero(value: torch.Tensor, padding: torch.Tensor) -> bool:
    """Return whether all sequence-padding positions are represented by zero."""

    if value.ndim < padding.ndim or tuple(value.shape[: padding.ndim]) != tuple(padding.shape):
        raise ValueError("V27d feature padding cannot be applied to tensor")
    mask = padding.reshape(*padding.shape, *([1] * (value.ndim - padding.ndim))).expand_as(value)
    return not bool(value.masked_select(mask).ne(0).any())


def feature_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d feature cache lacks tensors")
    metadata = {str(key): value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))})


def raw_scale_features(
    event_trajectory: torch.Tensor,
    padding: torch.Tensor,
    *,
    context: V27DContext,
) -> torch.Tensor:
    """Compute causal scale/update concentration facts from the raw trace."""

    if event_trajectory.ndim != 3 or event_trajectory.shape[-1] != _event_trajectory_dim(context):
        raise ValueError("V27d event trajectory width differs")
    if padding.shape != event_trajectory.shape[:2] or padding.dtype != torch.bool:
        raise ValueError("V27d raw-scale padding differs")
    raw_slices = _raw_slices(context.config.data.liquid_hidden_dim)
    values: list[torch.Tensor] = []
    for name in ("raw", "candidate", "mixed", "update"):
        norm = torch.linalg.vector_norm(event_trajectory[..., raw_slices[name]], dim=-1)
        values.append(torch.log1p(norm).unsqueeze(-1))
    update_norm = torch.linalg.vector_norm(event_trajectory[..., raw_slices["update"]], dim=-1)
    cumulative = update_norm.cumsum(dim=1).clamp_min(torch.finfo(update_norm.dtype).eps)
    values.append((update_norm / cumulative).unsqueeze(-1))
    result = torch.cat(values, dim=-1).masked_fill(padding.unsqueeze(-1), 0.0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("V27d raw-scale features are non-finite")
    return result


def stable8_causal_features(
    summary: torch.Tensor,
    reliable: torch.Tensor,
    reliable_count: torch.Tensor,
    reliable_fraction: torch.Tensor,
    padding: torch.Tensor,
    *,
    max_frames_per_event: int,
) -> torch.Tensor:
    """Transfer V27b-r6's reliability-aware stable8 statistics causally."""

    if summary.ndim != 3 or summary.shape[-1] != 8:
        raise ValueError("V27d stable8 summary width differs")
    if any(value.shape != summary.shape[:2] for value in (reliable, reliable_count, reliable_fraction, padding)):
        raise ValueError("V27d stable8 masks do not align")
    if reliable.dtype != torch.bool or reliable_count.dtype != torch.long or padding.dtype != torch.bool:
        raise ValueError("V27d stable8 mask dtype differs")
    if not reliable_fraction.is_floating_point() or not torch.isfinite(summary).all() or not torch.isfinite(reliable_fraction).all():
        raise ValueError("V27d stable8 values are invalid")
    active = ~padding
    valid = reliable & active
    weights = valid.to(dtype=summary.dtype).unsqueeze(-1)
    counts = weights.cumsum(dim=1)
    denominator = counts.clamp_min(1.0)
    total = (summary * weights).cumsum(dim=1)
    squared = (summary.square() * weights).cumsum(dim=1)
    mean = total / denominator
    std = (squared / denominator - mean.square()).clamp_min(0.0).sqrt()
    last = torch.zeros_like(summary)
    for index in range(summary.shape[1]):
        current = summary[:, index]
        if index == 0:
            last[:, index] = torch.where(valid[:, index].unsqueeze(-1), current, torch.zeros_like(current))
        else:
            last[:, index] = torch.where(valid[:, index].unsqueeze(-1), current, last[:, index - 1])
    prefix_events = active.to(dtype=summary.dtype).cumsum(dim=1).clamp_min(1.0)
    coverage = counts.squeeze(-1) / prefix_events
    observed_fraction = reliable_fraction * valid.to(dtype=summary.dtype)
    frame_fraction_mean = observed_fraction.cumsum(dim=1) / denominator.squeeze(-1)
    if isinstance(max_frames_per_event, bool) or int(max_frames_per_event) <= 0:
        raise ValueError("V27d stable8 maximum frame count is invalid")
    normalized_count = reliable_count.to(dtype=summary.dtype) / float(max_frames_per_event)
    frame_count_mean = (normalized_count * valid.to(dtype=summary.dtype)).cumsum(dim=1) / denominator.squeeze(-1)
    result = torch.cat(
        (
            last,
            mean,
            std,
            (last - mean).abs(),
            coverage.unsqueeze(-1),
            counts.clamp(max=float(summary.shape[1])) / float(summary.shape[1]),
            observed_fraction.unsqueeze(-1),
            frame_fraction_mean.unsqueeze(-1),
            (normalized_count * valid.to(dtype=summary.dtype)).unsqueeze(-1),
            frame_count_mean.unsqueeze(-1),
        ),
        dim=-1,
    )
    result = result.masked_fill(padding.unsqueeze(-1), 0.0)
    expected_width = summary.shape[-1] * 4 + 6
    if result.shape[-1] != expected_width or not torch.isfinite(result).all():
        raise FloatingPointError("V27d stable8 causal features are invalid")
    return result


def deterministic_quality_targets(
    *,
    task_facts: torch.Tensor,
    padding: torch.Tensor,
    context: V27DContext,
) -> torch.Tensor:
    if task_facts.ndim != 3 or padding.shape != task_facts.shape[:2] or padding.dtype != torch.bool:
        raise ValueError("V27d quality facts do not align")
    if task_facts.shape[-1] not in {context.config.data.l1_task_fact_dim, context.config.data.l2_task_fact_dim}:
        raise ValueError("V27d quality fact width differs")
    event_number = torch.arange(1, padding.shape[1] + 1, device=padding.device).unsqueeze(0)
    rd_fraction = task_facts[..., 1]
    dt_fraction = task_facts[..., 3]
    aux_fraction = task_facts[..., 7] if task_facts.shape[-1] == context.config.data.l1_task_fact_dim else torch.minimum(task_facts[..., 5], task_facts[..., 6])
    return (
        (~padding)
        & event_number.ge(context.config.calibration.minimum_quality_events)
        & rd_fraction.ge(context.config.calibration.minimum_quality_rd_fraction)
        & dt_fraction.ge(context.config.calibration.minimum_quality_dt_fraction)
        & aux_fraction.ge(context.config.calibration.minimum_quality_aux_fraction)
    )


def _parent_final(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, lengths - 1]


def _classification_correctness(
    labels: list[str],
    predictions: torch.Tensor,
    *,
    task: str,
) -> torch.Tensor:
    if predictions.ndim != 1 or predictions.shape[0] != len(labels):
        raise ValueError("V27d classification predictions do not align with labels")
    known_indices = {name: index for index, name in enumerate(V27D_KNOWN_LABELS)}
    l1_indices = {"ClutterTarget": 1, **{name: 0 for name in V27D_KNOWN_LABELS}}
    expected = []
    for row, label in enumerate(labels):
        if task == "known_gate":
            expected.append(label in l1_indices and int(predictions[row]) == l1_indices[label])
        elif task == "dbb_specialist":
            expected.append(label in known_indices and int(predictions[row]) == known_indices[label])
        else:
            raise ValueError(f"V27d classification task is unsupported: {task}")
    return torch.tensor(expected, dtype=torch.bool)


def _build_feature_payload(context: V27DContext, endpoint: Mapping[str, Any], *, variant: str, pool_record: Mapping[str, Any]) -> dict[str, Any]:
    if variant not in V27D_VARIANTS:
        raise ValueError(f"unsupported V27d feature variant: {variant}")
    tensors = endpoint["tensors"]
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    raw = raw_scale_features(tensors["event_trajectory"], padding, context=context)
    if raw.shape[-1] != context.config.model.raw_scale_feature_dim:
        raise ValueError("V27d raw-scale feature width differs from configuration")
    stable = stable8_causal_features(
        tensors["stable8_summary"],
        tensors["stable8_reliable"],
        tensors["stable8_reliable_count"],
        tensors["stable8_reliable_fraction"],
        padding,
        max_frames_per_event=context.config.data.max_frames_per_event,
    )
    if stable.shape[-1] != context.config.model.stable8_feature_dim:
        raise ValueError("V27d stable8 feature width differs from configuration")
    quality_targets = deterministic_quality_targets(task_facts=tensors["task_facts_sequence"], padding=padding, context=context)
    final_quality = _parent_final(quality_targets, lengths).to(dtype=torch.bool)
    task = str(endpoint["task"])
    labels = [str(value) for value in endpoint["label_names"]]
    if task == "known_gate":
        # L1 must protect both valid Target and valid Clutter actions from an
        # Unknown route.  Clutter is diagnostic-only for L2, not for L1.
        knownness_target = torch.tensor(
            [label != "UnknownTarget" for label in labels], dtype=torch.float32
        )
        knownness_population = torch.ones(len(labels), dtype=torch.bool)
        fit_policy = "L1 known Target/Clutter positive and Unknown negative; low-quality parents excluded"
    elif task == "dbb_specialist":
        knownness_target = torch.tensor(
            [label in V27D_KNOWN_LABELS for label in labels], dtype=torch.float32
        )
        knownness_population = torch.tensor(
            [label in {*V27D_KNOWN_LABELS, "UnknownTarget"} for label in labels],
            dtype=torch.bool,
        )
        fit_policy = "L2 DBB positive and Unknown negative; Clutter diagnostic only; low-quality parents excluded"
    else:
        raise ValueError(f"V27d feature cache task is unsupported: {task}")
    fit_eligible = final_quality & knownness_population
    final_endpoint_logits = _parent_final(tensors["endpoint_sequence_logits"], lengths)
    final_baseline_logits = _parent_final(tensors["baseline_sequence_logits"], lengths)
    class_prediction = final_endpoint_logits.argmax(dim=-1)
    class_correct = _classification_correctness(labels, class_prediction, task=task)
    tensor_payload = {
        "summary_features": tensors["summary_features"].clone(),
        "event_trajectory": tensors["event_trajectory"].clone(),
        "task_facts_sequence": tensors["task_facts_sequence"].clone(),
        "event_facts_sequence": tensors["event_facts_sequence"].clone(),
        "padding_mask": padding.clone(),
        "lengths": lengths.clone(),
        "baseline_sequence_logits": tensors["baseline_sequence_logits"].clone(),
        "endpoint_sequence_logits": tensors["endpoint_sequence_logits"].clone(),
        "endpoint_final_logits": final_endpoint_logits.clone(),
        "baseline_final_logits": final_baseline_logits.clone(),
        "raw_scale_features": raw.clone(),
        "stable8_causal_features": stable.clone(),
        "quality_target_sequence": quality_targets.clone(),
        "quality_target_final": final_quality,
        "knownness_target": knownness_target,
        "knownness_population": knownness_population,
        "fit_eligible": fit_eligible,
        "classification_prediction": class_prediction.to(dtype=torch.long),
        "classification_correct": class_correct,
        "stable8_reliable": tensors["stable8_reliable"].clone(),
        "stable8_reliable_count": tensors["stable8_reliable_count"].clone(),
        "stable8_reliable_fraction": tensors["stable8_reliable_fraction"].clone(),
        "rd_present": tensors["rd_present"].clone(),
    }
    if variant == "r1":
        tensor_payload["raw_scale_features"].zero_()
        tensor_payload["stable8_causal_features"].zero_()
    elif variant == "r2":
        tensor_payload["stable8_causal_features"].zero_()
    payload: dict[str, Any] = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "experiment_id": "V27d",
        "variant": variant,
        "task": task,
        "role": endpoint["role"],
        "endpoint_cache_sha256": endpoint["cache_sha256"],
        "endpoint_policy": endpoint["classification_endpoint"],
        "parent_ids": list(endpoint["parent_ids"]),
        "parent_ids_sha256": canonical_sha256(list(endpoint["parent_ids"])),
        "label_names": labels,
        "label_counts": dict(Counter(labels)),
        "fit_policy": fit_policy,
        "unknown_is_classifier_class": False,
        "unknown_classifier_training_exposure": 0,
        "test_payload_access_count": 0,
        "pool_parent_ids": list(pool_record["parent_ids"]),
        "tensors": tensor_payload,
    }
    payload["cache_sha256"] = feature_cache_digest(payload)
    return payload


def load_feature_manifest(context: V27DContext, *, variant: str) -> dict[str, Any]:
    path = artifact_path(context.project_root, f"src-v3/v27/V27d/cache/V27d-{variant}-feature-manifest.json")
    payload = read_sealed_json(path, expected_schema=FEATURE_MANIFEST_SCHEMA)
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    declared = body.pop("feature_manifest_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27d feature manifest hash differs")
    if payload.get("experiment_id") != "V27d" or payload.get("variant") != variant or payload.get("v27d_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27d feature manifest context differs")
    if payload.get("endpoint_manifest_sha256") != load_endpoint_manifest(context)["endpoint_manifest_sha256"]:
        raise ValueError("V27d feature manifest endpoint context differs")
    tasks = payload.get("tasks")
    expected_tasks = {f"{task}:{role}" for task in ("known_gate", "dbb_specialist") for role in V27D_ROLES}
    if not isinstance(tasks, Mapping) or set(tasks) != expected_tasks:
        raise ValueError("V27d feature manifest task set differs")
    for key, entry in tasks.items():
        task, role = key.split(":", 1)
        if not isinstance(entry, Mapping):
            raise ValueError(f"V27d feature manifest entry is invalid: {key}")
        expected_path = artifact_path(
            context.project_root,
            context.config.artifacts.feature_cache_template,
            variant=variant,
            task=task,
            role=role,
        )
        declared_path = project_path(context.project_root, str(entry.get("path", "")))
        if declared_path != expected_path:
            raise ValueError(f"V27d feature manifest path differs: {key}")
        for field in ("sha256", "cache_sha256"):
            value = str(entry.get(field, "")).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"V27d feature manifest {field} differs: {key}")
    return payload


def load_feature_cache(context: V27DContext, path: str | Path, *, verify_manifest: bool = True) -> dict[str, Any]:
    cache_path = project_path(context.project_root, path, must_exist=True)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d feature cache is not an object")
    value = dict(payload)
    declared = value.get("cache_sha256")
    if not isinstance(declared, str) or declared != feature_cache_digest(value):
        raise ValueError("V27d feature cache hash differs")
    if value.get("schema_version") != FEATURE_CACHE_SCHEMA or value.get("experiment_id") != "V27d":
        raise ValueError("V27d feature cache identity differs")
    if value.get("unknown_is_classifier_class") is not False or value.get("unknown_classifier_training_exposure") != 0:
        raise ValueError("V27d feature Unknown classifier contract differs")
    task = str(value.get("task"))
    if task not in {"known_gate", "dbb_specialist"} or str(value.get("variant")) not in V27D_VARIANTS or str(value.get("role")) not in V27D_ROLES:
        raise ValueError("V27d feature cache task/variant/role differs")
    parent_ids = [str(item) for item in value.get("parent_ids") or ()]
    if not parent_ids:
        raise ValueError("V27d feature cache cannot be empty")
    if value.get("parent_ids_sha256") != canonical_sha256(parent_ids) or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27d feature cache parent identity differs")
    if value.get("endpoint_cache_sha256") is None or not isinstance(value.get("endpoint_cache_sha256"), str):
        raise ValueError("V27d feature cache endpoint binding is missing")
    endpoint_target = artifact_path(
        context.project_root,
        context.config.artifacts.endpoint_cache_template,
        task=task,
        role=str(value["role"]),
    )
    endpoint_entry = _endpoint_manifest_entry(
        context,
        task=task,
        role=str(value["role"]),
        target=endpoint_target,
    )
    if value["endpoint_cache_sha256"] != endpoint_entry.get("cache_sha256"):
        raise ValueError("V27d feature cache endpoint binding differs")
    pool_parent_ids = [str(item) for item in value.get("pool_parent_ids") or ()]
    if pool_parent_ids != parent_ids:
        raise ValueError("V27d feature cache pool parent identity differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d feature cache tensors are missing")
    expected = {
        "summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence",
        "padding_mask", "lengths", "baseline_sequence_logits", "endpoint_sequence_logits", "endpoint_final_logits", "baseline_final_logits",
        "raw_scale_features", "stable8_causal_features", "quality_target_sequence", "quality_target_final",
        "knownness_target", "knownness_population", "fit_eligible", "classification_prediction", "classification_correct",
        "stable8_reliable", "stable8_reliable_count", "stable8_reliable_fraction", "rd_present",
    }
    if set(tensors) != expected:
        raise ValueError("V27d feature cache tensor fields differ")
    count = len(parent_ids)
    width = context.config.data.max_sequence_length
    task_facts = context.config.data.l1_task_fact_dim if task == "known_gate" else context.config.data.l2_task_fact_dim
    classes = context.config.data.l1_class_count if task == "known_gate" else context.config.data.l2_class_count
    shapes = {
        "summary_features": (count, width, _summary_feature_dim(context)), "event_trajectory": (count, width, _event_trajectory_dim(context)),
        "task_facts_sequence": (count, width, task_facts), "event_facts_sequence": (count, width, context.config.data.event_fact_dim),
        "padding_mask": (count, width), "lengths": (count,), "baseline_sequence_logits": (count, width, classes), "endpoint_sequence_logits": (count, width, classes),
        "endpoint_final_logits": (count, classes), "baseline_final_logits": (count, classes),
        "raw_scale_features": (count, width, context.config.model.raw_scale_feature_dim), "stable8_causal_features": (count, width, context.config.model.stable8_feature_dim),
        "quality_target_sequence": (count, width), "quality_target_final": (count,), "knownness_target": (count,),
        "knownness_population": (count,), "fit_eligible": (count,), "classification_prediction": (count,),
        "classification_correct": (count,), "stable8_reliable": (count, width),
        "stable8_reliable_count": (count, width), "stable8_reliable_fraction": (count, width), "rd_present": (count, width),
    }
    bool_fields = {"padding_mask", "quality_target_sequence", "quality_target_final", "knownness_population", "fit_eligible", "classification_correct", "stable8_reliable", "rd_present"}
    integer_fields = {"lengths", "classification_prediction", "stable8_reliable_count"}
    for name, shape in shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d feature tensor shape differs: {name}")
        if name in bool_fields and tensor.dtype != torch.bool:
            raise ValueError(f"V27d feature bool dtype differs: {name}")
        elif name in integer_fields and tensor.dtype != torch.long:
            raise ValueError(f"V27d feature integer dtype differs: {name}")
        elif name not in bool_fields and name not in integer_fields and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d feature values are invalid: {name}")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if bool((lengths < 1).any()) or bool((lengths > width).any()) or not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d feature lengths/padding differ")
    for name in ("summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "baseline_sequence_logits", "endpoint_sequence_logits", "raw_scale_features", "stable8_causal_features", "quality_target_sequence", "stable8_reliable", "stable8_reliable_count", "stable8_reliable_fraction", "rd_present"):
        value_tensor = tensors[name]
        if not _padded_values_are_zero(value_tensor, padding):
            raise ValueError(f"V27d feature padded value is non-zero: {name}")
    labels = [str(item) for item in value.get("label_names") or ()]
    if len(labels) != count or any(label not in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"} for label in labels):
        raise ValueError("V27d feature cache labels differ")
    if value.get("label_counts") != dict(Counter(labels)):
        raise ValueError("V27d feature cache label counts differ")
    population = tensors["knownness_population"]
    knownness_target = tensors["knownness_target"]
    quality = tensors["quality_target_final"]
    expected_population = (
        torch.ones(count, dtype=torch.bool)
        if task == "known_gate"
        else torch.tensor([label in {*V27D_KNOWN_LABELS, "UnknownTarget"} for label in labels], dtype=torch.bool)
    )
    expected_target = torch.tensor(
        [label != "UnknownTarget" if task == "known_gate" else label in V27D_KNOWN_LABELS for label in labels],
        dtype=knownness_target.dtype,
    )
    if knownness_target.dtype != torch.float32 or not torch.equal(population, expected_population) or not torch.equal(knownness_target, expected_target):
        raise ValueError("V27d feature knownness labels differ")
    if not torch.equal(tensors["fit_eligible"], quality & population):
        raise ValueError("V27d feature fit eligibility differs from quality population")
    if not torch.equal(tensors["quality_target_final"], _parent_final(tensors["quality_target_sequence"], lengths)):
        raise ValueError("V27d feature final quality differs from causal quality sequence")
    expected_quality = deterministic_quality_targets(
        task_facts=tensors["task_facts_sequence"],
        padding=padding,
        context=context,
    )
    if not torch.equal(tensors["quality_target_sequence"], expected_quality):
        raise ValueError("V27d feature quality targets differ from deterministic facts")
    expected_final_endpoint = _parent_final(tensors["endpoint_sequence_logits"], lengths)
    expected_final_baseline = _parent_final(tensors["baseline_sequence_logits"], lengths)
    if not torch.equal(tensors["endpoint_final_logits"], expected_final_endpoint):
        raise ValueError("V27d feature endpoint final logits differ from sequence")
    if not torch.equal(tensors["baseline_final_logits"], expected_final_baseline):
        raise ValueError("V27d feature baseline final logits differ from sequence")
    expected_prediction = tensors["endpoint_final_logits"].argmax(dim=-1)
    if not torch.equal(tensors["classification_prediction"], expected_prediction):
        raise ValueError("V27d feature classification prediction differs from endpoint logits")
    if not torch.equal(tensors["classification_correct"], _classification_correctness(labels, expected_prediction, task=task)):
        raise ValueError("V27d feature classification correctness differs from labels")
    if str(value["variant"]) == "r1":
        if bool(tensors["raw_scale_features"].ne(0).any()) or bool(tensors["stable8_causal_features"].ne(0).any()):
            raise ValueError("V27d r1 feature cache contains disabled feature variants")
    elif str(value["variant"]) == "r2" and bool(tensors["stable8_causal_features"].ne(0).any()):
        raise ValueError("V27d r2 feature cache contains disabled stable8 features")
    if verify_manifest:
        manifest = load_feature_manifest(context, variant=str(value["variant"]))
        entry = manifest["tasks"].get(f"{task}:{value['role']}")
        if not isinstance(entry, Mapping) or cache_path != project_path(context.project_root, str(entry.get("path", ""))):
            raise ValueError(f"V27d feature cache path is not canonical: {task}/{value['role']}")
        if sha256_file(cache_path) != entry.get("sha256") or declared != entry.get("cache_sha256"):
            raise ValueError(f"V27d feature cache manifest binding differs: {task}/{value['role']}")
    return {**value, "tensors": {str(name): tensor for name, tensor in tensors.items()}}


def materialize_feature_caches(context: V27DContext, *, variant: str) -> dict[str, Any]:
    if variant not in V27D_VARIANTS:
        raise ValueError(f"unsupported V27d feature variant: {variant}")
    pools = load_pools(context)
    endpoint_manifest = load_endpoint_manifest(context)
    results: dict[str, Any] = {}
    for task in ("known_gate", "dbb_specialist"):
        for role in V27D_ROLES:
            entry = endpoint_manifest["tasks"][f"{task}:{role}"]
            endpoint_path = project_path(context.project_root, str(entry["path"]), must_exist=True)
            endpoint = load_endpoint_cache(context, path=endpoint_path, pools=pools, task=task, role=role)
            payload = _build_feature_payload(context, endpoint, variant=variant, pool_record=pools["roles"][role])
            path = artifact_path(context.project_root, context.config.artifacts.feature_cache_template, variant=variant, role=role, task=task)
            write_torch_if_absent_or_identical(path, payload)
            checked = load_feature_cache(context, path, verify_manifest=False)
            results[f"{task}:{role}"] = {
                "path": relative_project_path(context.project_root, path),
                "sha256": sha256_file(path),
                "cache_sha256": checked["cache_sha256"],
                "parent_count": len(payload["parent_ids"]),
                "label_counts": payload["label_counts"],
                "fit_eligible_count": int(payload["tensors"]["fit_eligible"].sum().item()),
                "quality_eligible_count": int(payload["tensors"]["quality_target_final"].sum().item()),
            }
    manifest: dict[str, Any] = {
        "schema_version": FEATURE_MANIFEST_SCHEMA,
        "experiment_id": "V27d",
        "status": "prepared",
        "variant": variant,
        "v27d_manifest_sha256": context.manifest_sha256,
        "endpoint_manifest_sha256": endpoint_manifest["endpoint_manifest_sha256"],
        "tasks": results,
        "test_payload_access_count": 0,
    }
    manifest["feature_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = artifact_path(context.project_root, f"src-v3/v27/V27d/cache/V27d-{variant}-feature-manifest.json")
    from .protocol import write_sealed_json

    write_sealed_json(manifest_path, manifest)
    for task in ("known_gate", "dbb_specialist"):
        for role in V27D_ROLES:
            path = artifact_path(
                context.project_root,
                context.config.artifacts.feature_cache_template,
                variant=variant,
                role=role,
                task=task,
            )
            load_feature_cache(context, path)
    return {**manifest, "manifest_path": str(manifest_path), "manifest_file_sha256": sha256_file(manifest_path)}


__all__ = [
    "FEATURE_CACHE_SCHEMA",
    "FEATURE_MANIFEST_SCHEMA",
    "deterministic_quality_targets",
    "feature_cache_digest",
    "load_feature_manifest",
    "load_feature_cache",
    "materialize_feature_caches",
    "raw_scale_features",
    "stable8_causal_features",
]
