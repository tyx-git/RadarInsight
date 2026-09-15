"""Materialize and verify frozen V28 auxiliary-head representations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from .checkpoint import load_checkpoint
from .data import V28RoleDataset, build_dataloader, model_input_kwargs
from .heads import V28TrajectoryHeads
from .model import V28TaskModel
from .pools import load_pool_registry
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, V28_REPRESENTATION_SCHEMA, V28_ROLES, artifact_path


def _tensor_hash(values: Mapping[str, torch.Tensor]) -> str:
    records: list[Any] = []
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        records.append((name, str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes().hex()))
    return canonical_sha256(records)


def representation_path(context: V28Context, *, task: str, role: str) -> Path:
    return artifact_path(context.project_root, str(context.config.artifacts["representation_template"]), task=task, role=role, seed=0)


def _endpoint_binding(context: V28Context, *, task: str, checkpoint: Path) -> dict[str, Any]:
    return {"task": task, "path": relative_project_path(context.project_root, checkpoint), "sha256": sha256_file(checkpoint)}


def _concat_sequence_batches(arrays: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate variable-length batches under one canonical time width."""

    if not arrays or any(not values for values in arrays.values()):
        raise ValueError("V28 representation batches are empty")
    max_length = max(int(value.shape[1]) for value in arrays["knownness_features"])
    result: dict[str, torch.Tensor] = {}
    for name, values in arrays.items():
        if name == "lengths":
            result[name] = torch.cat(values, dim=0)
            continue
        padded: list[torch.Tensor] = []
        for value in values:
            if value.ndim < 2 or int(value.shape[1]) > max_length:
                raise ValueError(f"V28 representation batch shape differs: {name}")
            width = max_length - int(value.shape[1])
            if width:
                shape = (int(value.shape[0]), width, *value.shape[2:])
                fill = True if name == "padding_mask" else 0
                suffix = torch.full(shape, fill, dtype=value.dtype, device=value.device)
                value = torch.cat((value, suffix), dim=1)
            padded.append(value)
        result[name] = torch.cat(padded, dim=0)
    return result


def _write_representation(path: Path, payload: Mapping[str, Any], *, context: V28Context) -> None:
    """Archive a stale manifest-bound cache before creating its replacement."""

    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(existing, Mapping) and existing.get("system_manifest_sha256") != context.manifest_sha256:
            archive = path.parent / "archive"
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / f"{path.name}.stale-before-{sha256_file(path)[:16]}"
            if destination.exists():
                if sha256_file(destination) != sha256_file(path):
                    raise FileExistsError(f"V28 representation archive collision: {destination}")
                path.unlink()
            else:
                path.replace(destination)
    write_torch_if_absent_or_identical(path, payload)


def materialize_representations(context: V28Context, *, endpoint_binding: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    pool = load_pool_registry(context, endpoint_binding=endpoint_binding)
    results: dict[str, Any] = {}
    for task in (V28_L1_TASK, V28_L2_TASK):
        endpoint = endpoint_binding[task]
        model = V28TaskModel(context, task=task).to(device)
        load_checkpoint(context, model, endpoint["path"], task=task, seed=int(endpoint["seed"]), map_location=device)
        model.eval()
        heads = V28TrajectoryHeads(context, task=task).to(device)
        for role in V28_ROLES:
            entry = pool["roles"][task][role]
            dataset = V28RoleDataset(context, task=task, role=role, parent_ids=entry["parent_ids"], split="validation" if role == "calibration" else "train")
            arrays: dict[str, list[torch.Tensor]] = {"knownness_features": [], "quality_features": [], "padding_mask": [], "lengths": [], "sequence_logits": []}
            labels: list[str] = []
            parent_ids: list[str] = []
            with torch.inference_mode():
                for batch in build_dataloader(dataset, batch_size=int(context.config.training["batch_size"]), shuffle=False, seed=int(context.config.training["dataloader_seed"])):
                    output = model(**model_input_kwargs(batch, device))
                    features = heads.build_features(output)
                    arrays["knownness_features"].append(features["knownness_features"].cpu())
                    arrays["quality_features"].append(features["quality_features"].cpu())
                    arrays["padding_mask"].append(batch["tr_padding_mask"].cpu())
                    arrays["lengths"].append(batch["lengths"].cpu())
                    arrays["sequence_logits"].append(output["sequence_logits"].cpu())
                    labels.extend(str(value) for value in batch["base_label_names"])
                    parent_ids.extend(str(value) for value in batch["parent_sample_ids"])
            tensors = _concat_sequence_batches(arrays)
            if parent_ids != [str(value) for value in entry["parent_ids"]] or labels != [str(dataset.base._groups[value].iloc[0]["label_name"]) for value in entry["parent_ids"]]:
                raise ValueError(f"V28 representation parent order differs: {task}/{role}")
            padding = tensors["padding_mask"]
            for name in ("knownness_features", "quality_features", "sequence_logits"):
                if bool(tensors[name].masked_select(padding.unsqueeze(-1)).ne(0).any()):
                    raise ValueError(f"V28 padded representation values are non-zero: {task}/{role}/{name}")
            metadata = {"schema_version": V28_REPRESENTATION_SCHEMA, "experiment_id": "V28", "status": "prepared", "task": task, "role": role, "system_manifest_sha256": context.manifest_sha256, "pool_registry_sha256": pool["registry_sha256"], "endpoint_binding": dict(endpoint), "parent_ids": parent_ids, "parent_ids_sha256": canonical_sha256(parent_ids), "label_names": labels, "label_counts": dict(Counter(labels)), "test_payload_access_count": 0}
            payload = {**metadata, "tensors": tensors}
            payload["cache_sha256"] = canonical_sha256({"metadata": {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": _tensor_hash(tensors)})
            path = representation_path(context, task=task, role=role)
            _write_representation(path, payload, context=context)
            results[f"{task}:{role}"] = {"path": relative_project_path(context.project_root, path), "sha256": sha256_file(path), "cache_sha256": payload["cache_sha256"], "parent_count": len(parent_ids), "label_counts": dict(Counter(labels))}
    manifest = {"schema_version": "RADAR-LN3-V28-representation-manifest-v2", "experiment_id": "V28", "status": "prepared", "system_manifest_sha256": context.manifest_sha256, "pool_registry_sha256": pool["registry_sha256"], "tasks": results, "test_payload_access_count": 0}
    manifest["representation_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = artifact_path(context.project_root, str(context.config.artifacts["representation_manifest_path"]))
    from .protocol import write_sealed_json
    write_sealed_json(manifest_path, manifest)
    return {"manifest_path": relative_project_path(context.project_root, manifest_path), "manifest_sha256": sha256_file(manifest_path), "tasks": results}


def load_representation(context: V28Context, *, task: str, role: str) -> dict[str, Any]:
    path = representation_path(context, task=task, role=role)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != V28_REPRESENTATION_SCHEMA or payload.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError(f"V28 representation identity differs: {task}/{role}")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V28 representation tensors are missing")
    digest = canonical_sha256({"metadata": {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": _tensor_hash(tensors)})
    if payload.get("cache_sha256") != digest:
        raise ValueError(f"V28 representation hash differs: {task}/{role}")
    return dict(payload)


__all__ = ["load_representation", "materialize_representations", "representation_path"]
