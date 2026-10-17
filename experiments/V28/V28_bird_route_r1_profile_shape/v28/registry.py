"""Hash-bound V28 L3 registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, sha256_file

from .pools import verify_pool_registry
from .protocol import V28Context, V28_HEAD_SCHEMA, V28_L3_SCHEMA, V28_L1_TASK, V28_L2_TASK, artifact_path, checkpoint_manifest_compatibility, read_sealed_json, resolve_project_path


_CLASSIFIER_FEATURE_MANIFEST = Path("src-v3/v28/calibration/V28-classifier-feature-manifest-v2.json")


def _verify_custom_digest(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = dict(payload)
    declared = value.pop(field, None)
    value.pop("payload_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(value):
        raise ValueError(f"V28 {label} digest differs")
    return declared


def _normalization_contract(context: V28Context) -> dict[str, Any]:
    data = context.config.data
    model = context.config.model
    return {
        "index_schema_version": str(data["index_schema_version"]),
        "max_sequence_length": int(data["max_sequence_length"]),
        "max_rd_frames_per_point": int(data["max_rd_frames_per_point"]),
        "rd_shape": {"height": int(model["rd_height"]), "width": int(model["rd_width"]), "channels": 2},
        "l1": {
            "course_feature_name": str(data["l1"]["course_feature_name"]),
            "course_feature_variant": str(data["l1"]["course_feature_variant"]),
            "course_minimum_valid_nonfirst_points": int(data["l1"]["course_minimum_valid_nonfirst_points"]),
            "rd_amplitude_center": float(data["l1"]["rd_amplitude_center"]),
            "rd_amplitude_scale": float(data["l1"]["rd_amplitude_scale"]),
        },
        "l2": {
            "rd_amplitude_center": float(data["l2"]["rd_amplitude_center"]),
            "rd_amplitude_scale": float(data["l2"]["rd_amplitude_scale"]),
            "tr_clipping": dict(data["l2"]["tr_clipping"]),
        },
    }


def _verify_classifier_feature_manifest(context: V28Context) -> tuple[Path, dict[str, Any]]:
    path = resolve_project_path(context.project_root, _CLASSIFIER_FEATURE_MANIFEST, must_exist=True)
    payload = read_sealed_json(path)
    if payload.get("schema_version") != "RADAR-LN3-V28-classifier-feature-manifest-v2" or payload.get("status") != "prepared":
        raise ValueError("V28 classifier feature manifest identity differs")
    if payload.get("config_sha256") != context.config.sha256 or payload.get("index_sha256") != sha256_file(context.project_root / context.config.data["index_path"]):
        raise ValueError("V28 classifier feature manifest provenance differs")
    if payload.get("unknown_exposure") != 0 or int(payload.get("test_payload_access_count", -1)) != 0:
        raise ValueError("V28 classifier feature manifest violated the test/Unknown boundary")
    _verify_custom_digest(payload, "feature_manifest_sha256", "classifier feature manifest")
    for key, record in (payload.get("caches") or {}).items():
        if not isinstance(record, Mapping):
            raise ValueError(f"V28 classifier feature cache record is invalid: {key}")
        if "path" in record:
            cache_path = resolve_project_path(context.project_root, str(record["path"]), must_exist=True)
            if sha256_file(cache_path) != record.get("sha256"):
                raise ValueError(f"V28 classifier feature cache hash differs: {key}")
        elif key.startswith(f"{V28_L1_TASK}:") and record.get("unknown_exposure") == 0:
            continue
        else:
            raise ValueError(f"V28 classifier feature cache binding is missing: {key}")
    return path, payload


def registry_path(context: V28Context) -> Path:
    return artifact_path(context.project_root, str(context.config.artifacts["registry_path"]))


def _verify_binding_file(context: V28Context, task: str, binding: Mapping[str, Any], *, head: bool) -> Path:
    path = resolve_project_path(context.project_root, str(binding.get("path")), must_exist=True)
    if sha256_file(path) != binding.get("sha256") or binding.get("task") != task:
        raise ValueError(f"V28 registry binding file differs: {task}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("task") != task or int(payload.get("seed", -1)) != int(binding.get("seed", -2)):
        raise ValueError(f"V28 registry binding metadata differs: {task}")
    if head:
        if payload.get("schema_version") != V28_HEAD_SCHEMA:
            raise ValueError(f"V28 registry head schema differs: {task}")
        from .heads import V28TrajectoryHeads
        from .head_training import _load_head_payload
        payload = _load_head_payload(path)
        expected_contract = V28TrajectoryHeads(context, task=task).model_contract()
        if payload.get("model_contract") != expected_contract or payload.get("model_contract_sha256") != canonical_sha256(expected_contract):
            raise ValueError(f"V28 registry head contract differs: {task}")
        trained_manifest = payload.get("system_manifest_sha256")
    else:
        if payload.get("schema_version") != "RADAR-LN3-V28-task-checkpoint-v2":
            raise ValueError(f"V28 registry endpoint schema differs: {task}")
        trained_manifest = payload.get("manifest_sha256")
    if trained_manifest != binding.get("trained_manifest_sha256"):
        raise ValueError(f"V28 registry trained manifest binding differs: {task}")
    if trained_manifest != context.manifest_sha256:
        checkpoint_manifest_compatibility(
            context.project_root,
            {**context.manifest, "_manifest_path": context.manifest_path.relative_to(context.project_root).as_posix()},
            str(trained_manifest),
            task=task,
        )
    return path


def _verify_representation_manifest(
    context: V28Context,
    path: Path,
    *,
    endpoint_binding: Mapping[str, Any],
    head_bindings: Mapping[str, Any],
    pool_registry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_sealed_json(path)
    if payload.get("schema_version") != "RADAR-LN3-V28-representation-manifest-v2" or payload.get("status") != "prepared":
        raise ValueError("V28 representation manifest identity differs")
    _verify_custom_digest(payload, "representation_manifest_sha256", "representation manifest")
    if payload.get("system_manifest_sha256") != context.manifest_sha256 or payload.get("pool_registry_sha256") != pool_registry.get("registry_sha256"):
        raise ValueError("V28 representation manifest provenance differs")
    expected_keys = {f"{task}:{role}" for task in (V28_L1_TASK, V28_L2_TASK) for role in ("representation_fit", "representation_development", "calibration")}
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != expected_keys:
        raise ValueError("V28 representation manifest task coverage differs")
    cache_payloads: dict[str, Mapping[str, Any]] = {}
    for key, record in tasks.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"V28 representation manifest record is invalid: {key}")
        cache_path = resolve_project_path(context.project_root, str(record.get("path")), must_exist=True)
        if sha256_file(cache_path) != record.get("sha256"):
            raise ValueError(f"V28 representation cache hash differs: {key}")
        task, role = key.split(":", 1)
        # The sealed representation manifest already records the cache's
        # content hash and provenance.  Rechecking the file SHA-256 here is
        # sufficient; loading six 20-48 MB tensor payloads during every
        # registry verification only duplicates memory and can stall a
        # Windows run without adding provenance coverage.
        if (
            not record.get("cache_sha256")
            or not record.get("sha256")
            or record.get("task") != task
            or record.get("role") != role
            or record.get("system_manifest_sha256") != context.manifest_sha256
            or record.get("pool_registry_sha256") != pool_registry.get("registry_sha256")
            or dict(record.get("endpoint_binding") or {}) != dict(endpoint_binding.get(task) or {})
        ):
            raise ValueError(f"V28 representation cache provenance differs: {key}")
        cache_payloads[key] = record
    for task, binding in head_bindings.items():
        head_path = _verify_binding_file(context, task, binding, head=True)
        head = torch.load(head_path, map_location="cpu", weights_only=False)
        for role, field in (("representation_fit", "representation_fit_cache_sha256"), ("representation_development", "representation_development_cache_sha256")):
            if head.get(field) != cache_payloads[f"{task}:{role}"].get("cache_sha256"):
                raise ValueError(f"V28 head representation binding differs: {task}/{role}")
    return payload


def _verify_calibration_representation_binding(
    context: V28Context,
    calibration: Mapping[str, Any],
    representation_manifest: Mapping[str, Any],
) -> None:
    tasks = representation_manifest.get("tasks")
    calibration_tasks = calibration.get("tasks")
    if not isinstance(tasks, Mapping) or not isinstance(calibration_tasks, Mapping):
        raise ValueError("V28 calibration/representation task sections are invalid")
    for task in (V28_L1_TASK, V28_L2_TASK):
        expected = tasks.get(f"{task}:calibration")
        observed = calibration_tasks.get(task, {}).get("representation_calibration")
        if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
            raise ValueError(f"V28 calibration representation binding is missing: {task}")
        if observed.get("path") != expected.get("path") or observed.get("file_sha256") != expected.get("sha256") or observed.get("cache_sha256") != expected.get("cache_sha256"):
            raise ValueError(f"V28 calibration representation binding differs: {task}")
        path = resolve_project_path(context.project_root, str(observed.get("path")), must_exist=True)
        if sha256_file(path) != observed.get("file_sha256"):
            raise ValueError(f"V28 calibration representation file hash differs: {task}")






def verify_registry(context: V28Context, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    envelope = value.pop("payload_sha256", None)
    if envelope != canonical_sha256(value):
        raise ValueError("V28 L3 registry envelope hash differs")
    declared = value.pop("registry_sha256", None)
    if declared != canonical_sha256(value):
        raise ValueError("V28 L3 registry hash differs")
    if value.get("schema_version") != V28_L3_SCHEMA or value.get("experiment_id") != "V28" or value.get("system_manifest_sha256") != context.manifest_sha256 or value.get("config_sha256") != context.config.sha256:
        raise ValueError("V28 L3 registry identity differs")
    status = str(value.get("status", "unavailable"))
    if status not in {"fitted", "internal_exploratory", "unavailable"}:
        raise ValueError("V28 L3 registry status is invalid")
    if value.get("action_contract") != {"unknown_is_classifier_class": False, "pending_is_not_unknown_rejection": True, "quality_first": True, "registry_fail_closed": True, "fixed_softmax_threshold": False, "single_global_distance_gate": False}:
        raise ValueError("V28 L3 action contract differs")
    if int(value.get("test_payload_access_count", -1)) != 0:
        raise ValueError("V28 L3 registry violated test boundary")
    for section in ("endpoint_binding", "head_bindings"):
        bindings = value.get(section)
        if not isinstance(bindings, Mapping) or set(bindings) != {V28_L1_TASK, V28_L2_TASK}:
            raise ValueError(f"V28 L3 registry {section} is incomplete")
        for task, binding in bindings.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"V28 L3 registry {section} binding is invalid: {task}")
            _verify_binding_file(context, task, binding, head=section == "head_bindings")
    pool = value.get("pool_registry")
    if not isinstance(pool, Mapping):
        raise ValueError("V28 L3 registry pool binding is missing")
    pool_path = resolve_project_path(context.project_root, str(pool.get("path")), must_exist=True)
    if sha256_file(pool_path) != pool.get("sha256"):
        raise ValueError("V28 L3 registry pool hash differs")
    pool_payload = verify_pool_registry(context, read_sealed_json(pool_path), endpoint_binding=value["endpoint_binding"])
    classifier_feature = value.get("classifier_feature_manifest")
    if not isinstance(classifier_feature, Mapping):
        raise ValueError("V28 L3 registry classifier feature binding is missing")
    classifier_feature_path = resolve_project_path(context.project_root, str(classifier_feature.get("path")), must_exist=True)
    if sha256_file(classifier_feature_path) != classifier_feature.get("sha256"):
        raise ValueError("V28 L3 registry classifier feature hash differs")
    checked_classifier_feature_path, checked_classifier_feature = _verify_classifier_feature_manifest(context)
    if classifier_feature_path != checked_classifier_feature_path or classifier_feature.get("feature_manifest_sha256") != checked_classifier_feature.get("feature_manifest_sha256"):
        raise ValueError("V28 L3 registry classifier feature provenance differs")
    if value.get("normalization_contract") != _normalization_contract(context):
        raise ValueError("V28 L3 registry normalization contract differs")
    feature = value.get("feature_manifest")
    if feature is not None:
        if not isinstance(feature, Mapping):
            raise ValueError("V28 L3 registry feature binding is invalid")
        feature_path = resolve_project_path(context.project_root, str(feature.get("path")), must_exist=True)
        if sha256_file(feature_path) != feature.get("sha256"):
            raise ValueError("V28 L3 registry feature hash differs")
        feature_payload = read_sealed_json(feature_path)
        _verify_custom_digest(feature_payload, "feature_manifest_sha256", "open-set feature manifest")
        if feature_payload.get("system_manifest_sha256") != context.manifest_sha256 or feature_payload.get("pool_registry_sha256") != pool_payload.get("registry_sha256") or feature_payload.get("endpoint_binding") != value.get("endpoint_binding"):
            raise ValueError("V28 L3 registry feature provenance differs")
        for record in (feature_payload.get("caches") or {}).values():
            cache_path = resolve_project_path(context.project_root, str(record.get("path")), must_exist=True)
            if sha256_file(cache_path) != record.get("sha256"):
                raise ValueError("V28 L3 registry feature cache hash differs")
    representation = value.get("representation_manifest")
    if not isinstance(representation, Mapping):
        raise ValueError("V28 L3 registry representation manifest binding is missing")
    representation_path = resolve_project_path(context.project_root, str(representation.get("path")), must_exist=True)
    if sha256_file(representation_path) != representation.get("sha256"):
        raise ValueError("V28 L3 registry representation manifest hash differs")
    representation_payload = read_sealed_json(representation_path)
    _verify_custom_digest(representation_payload, "representation_manifest_sha256", "representation manifest")
    if representation_payload.get("payload_sha256") != representation.get("payload_sha256"):
        raise ValueError("V28 L3 registry representation manifest payload hash differs")
    report = value.get("calibration_report")
    if not isinstance(report, Mapping):
        raise ValueError("V28 L3 registry calibration report binding is missing")
    report_path = resolve_project_path(context.project_root, str(report.get("path")), must_exist=True)
    if sha256_file(report_path) != report.get("sha256"):
        raise ValueError("V28 L3 registry calibration report hash differs")
    calibration = read_sealed_json(report_path)
    if calibration.get("payload_sha256") != value.get("calibration_payload_sha256") or calibration.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V28 L3 registry calibration binding differs")
    if value.get("calibration") != calibration or value.get("status") != calibration.get("status") or value.get("deployment_eligible") != bool(status == "fitted" and calibration.get("deployment_eligible") is True):
        raise ValueError("V28 L3 registry calibration status or payload differs")
    if value.get("calibration", {}).get("endpoint_binding") != value.get("endpoint_binding") or value.get("calibration", {}).get("head_bindings") != value.get("head_bindings"):
        raise ValueError("V28 L3 registry calibration endpoint binding differs")
    representation_payload = _verify_representation_manifest(context, representation_path, endpoint_binding=value["endpoint_binding"], head_bindings=value["head_bindings"], pool_registry=pool_payload)
    _verify_calibration_representation_binding(context, calibration, representation_payload)
    return {**value, "registry_sha256": declared, "payload_sha256": envelope}


def load_registry(context: V28Context) -> dict[str, Any]:
    return verify_registry(context, read_sealed_json(registry_path(context)))


__all__ = ["load_registry", "registry_path", "verify_registry"]