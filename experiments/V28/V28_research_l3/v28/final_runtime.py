"""Hash-bound loaders for the final V28 research endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file

from .final_heads import FinalL2ProfileResidualHead, FinalTemporalKnownnessHead
from .protocol import V28Context, V28_L1_TASK, V28_L2_TASK, checkpoint_manifest_compatibility


FINAL_ROOT = Path("src-v3/v28/final")
HEADS_ROOT = FINAL_ROOT / "heads"
FEATURES_ROOT = FINAL_ROOT / "features"
CALIBRATION_ROOT = FINAL_ROOT / "calibration-candidates"
NORM_HEAD_SCHEMA = "RADAR-LN3-V28-knownness-temporal-norm-head-v3"
PROFILE_HEAD_SCHEMA = "RADAR-LN3-V28-knownness-l2-profile-head-v4"


def final_path(root: Path, relative: Path) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve() / FINAL_ROOT)
    except ValueError as exc:
        raise ValueError(f"V28 final path escaped final bundle: {relative}") from exc
    return value


def artifact(root: Path, path: Path) -> dict[str, str]:
    return {"path": relative_project_path(root, path), "sha256": sha256_file(path)}


def tensor_hash(values: Mapping[str, torch.Tensor]) -> str:
    records = []
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        records.append((name, str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes().hex()))
    return canonical_sha256(records)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tensors"), Mapping):
        raise ValueError(f"V28 final tensor payload is invalid: {path}")
    value = dict(payload)
    digest = canonical_sha256({"metadata": {key: item for key, item in value.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": tensor_hash(value["tensors"])})
    if value.get("cache_sha256") != digest:
        raise ValueError(f"V28 final tensor payload hash differs: {path}")
    return value


def state_dict_hash(state: Any) -> str:
    if not isinstance(state, Mapping) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("V28 final head state is invalid")
    return canonical_sha256({key: value.detach().cpu().numpy().tobytes().hex() for key, value in sorted(state.items())})


def _verify_manifest_compatibility(context: V28Context, payload: Mapping[str, Any], task: str) -> None:
    trained = str(payload.get("system_manifest_sha256"))
    if trained == context.manifest_sha256:
        return
    checkpoint_manifest_compatibility(
        context.project_root,
        {**context.manifest, "_manifest_path": context.manifest_path.relative_to(context.project_root).as_posix()},
        trained,
        task=task,
    )


def _load_norm_head(context: V28Context, *, task: str, seed: int) -> tuple[torch.nn.Module, dict[str, Any], Path, dict[str, Any]]:
    path = final_path(context.project_root, HEADS_ROOT / f"{task}-treatment-seed{seed}.pt" if task == V28_L1_TASK else HEADS_ROOT / "l2_dbb_specialist-temporal-norm-treatment-seed43.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != NORM_HEAD_SCHEMA or payload.get("task") != task or payload.get("variant") != "treatment" or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"V28 final temporal head identity differs: {task}")
    if payload.get("state_dict_sha256") != state_dict_hash(payload.get("state_dict")):
        raise ValueError(f"V28 final temporal head provenance differs: {task}")
    _verify_manifest_compatibility(context, payload, task)
    contract = payload.get("model_contract")
    if payload.get("model_contract_sha256") != canonical_sha256(contract) or not isinstance(contract, Mapping) or contract.get("task") != task:
        raise ValueError(f"V28 final temporal head contract differs: {task}")
    feature = load_payload(final_path(context.project_root, FEATURES_ROOT / f"representation_development-{task}-treatment.pt"))
    feature_source = payload.get("feature_source")
    if not isinstance(feature_source, Mapping) or feature_source.get("development_cache_sha256") != feature.get("cache_sha256"):
        raise ValueError(f"V28 final temporal head feature binding differs: {task}")
    tensors = feature["tensors"]
    head = FinalTemporalKnownnessHead(context, task=task, input_dim=int(tensors["knownness_features"].shape[-1]), temporal_dim=int(tensors["temporal_features"].shape[-1]), quality_input_dim=int(tensors["quality_features"].shape[-1]))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return head, dict(payload), path, feature


def load_final_l1(context: V28Context) -> dict[str, Any]:
    head, payload, path, feature = _load_norm_head(context, task=V28_L1_TASK, seed=42)
    return {"task": V28_L1_TASK, "head": head, "payload": payload, "path": path, "feature": feature, "composition": {"type": "direct", "head": artifact(context.project_root, path)}}


def load_profile_payload(context: V28Context, *, role: str) -> dict[str, Any]:
    return load_payload(final_path(context.project_root, FEATURES_ROOT / f"representation_{role.replace('representation_', '')}-l2-profile-treatment.pt"))


def load_final_l2(context: V28Context) -> dict[str, Any]:
    base, base_payload, base_path, feature = _load_norm_head(context, task=V28_L2_TASK, seed=43)
    residual_path = final_path(context.project_root, HEADS_ROOT / "l2_dbb_specialist-profile-residual-treatment-seed43.pt")
    residual_payload = torch.load(residual_path, map_location="cpu", weights_only=False)
    if not isinstance(residual_payload, Mapping) or residual_payload.get("schema_version") != PROFILE_HEAD_SCHEMA or residual_payload.get("task") != V28_L2_TASK or int(residual_payload.get("seed", -1)) != 43:
        raise ValueError("V28 final L2 profile head identity differs")
    if residual_payload.get("state_dict_sha256") != state_dict_hash(residual_payload.get("state_dict")):
        raise ValueError("V28 final L2 profile head provenance differs")
    _verify_manifest_compatibility(context, residual_payload, V28_L2_TASK)
    profile_fit = load_profile_payload(context, role="fit")
    profile_development = load_profile_payload(context, role="development")
    if residual_payload.get("fit_cache_sha256") != profile_fit.get("cache_sha256") or residual_payload.get("development_cache_sha256") != profile_development.get("cache_sha256"):
        raise ValueError("V28 final L2 profile representation binding differs")
    baseline = residual_payload.get("baseline_head")
    if not isinstance(baseline, Mapping) or baseline.get("path") not in {relative_project_path(context.project_root, base_path), relative_project_path(context.project_root, HEADS_ROOT / "l2_dbb_specialist-temporal-norm-treatment-seed43.pt")} or baseline.get("sha256") != sha256_file(base_path) or residual_payload.get("baseline_head_state_dict_sha256") != base_payload.get("state_dict_sha256"):
        raise ValueError("V28 final L2 base head binding differs")
    profile_dim = int(profile_development["tensors"]["profile_features"].shape[-1])
    contract = residual_payload.get("model_contract")
    if not isinstance(contract, Mapping) or int(contract.get("profile_feature_dim", -1)) != profile_dim or residual_payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V28 final L2 profile contract differs")
    head = FinalL2ProfileResidualHead(base, profile_dim=profile_dim)
    head.profile_branch.load_state_dict(residual_payload["state_dict"], strict=True)
    head.eval()
    return {"task": V28_L2_TASK, "head": head, "payload": dict(residual_payload), "base_payload": base_payload, "base_path": base_path, "residual_path": residual_path, "feature": feature, "profile_development": profile_development, "composition": {"type": "frozen_base_plus_profile_residual", "baseline": artifact(context.project_root, base_path), "residual": artifact(context.project_root, residual_path), "baseline_state_dict_sha256": base_payload.get("state_dict_sha256"), "residual_state_dict_sha256": residual_payload.get("state_dict_sha256"), "model_contract": contract, "model_contract_sha256": residual_payload.get("model_contract_sha256")}}


__all__ = ["FINAL_ROOT", "HEADS_ROOT", "FEATURES_ROOT", "CALIBRATION_ROOT", "final_path", "artifact", "load_payload", "tensor_hash", "state_dict_hash", "load_final_l1", "load_profile_payload", "load_final_l2"]
