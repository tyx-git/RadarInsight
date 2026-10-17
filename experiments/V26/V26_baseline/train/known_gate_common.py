from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA, KNOWN_GATE_BASELINE_MULTISEED_SEEDS, KNOWN_GATE_CLASS_NAMES, KNOWN_GATE_CONTRACT_SCHEMA, KNOWN_GATE_MANIFEST_SCHEMA, KNOWN_GATE_READINESS_SCHEMA, KNOWN_GATE_SELECTION_METRIC, KNOWN_GATE_V21E_MANIFEST_SCHEMA, KNOWN_GATE_V24_EXPERIMENT_SCHEMA, KNOWN_GATE_V24_MANIFEST_SCHEMA, KNOWN_GATE_V24A_EXPERIMENT_SCHEMA, KNOWN_GATE_V24A_MANIFEST_SCHEMA, KNOWN_GATE_V24C_EXPERIMENT_SCHEMA, KNOWN_GATE_V24C_MANIFEST_SCHEMA, KNOWN_GATE_V25A_EXPERIMENT_SCHEMA, KNOWN_GATE_V25A_MANIFEST_SCHEMA, KNOWN_GATE_V25D_EXPERIMENT_SCHEMA, KNOWN_GATE_V25D_MANIFEST_SCHEMA, KNOWN_GATE_V25E_EXPERIMENT_SCHEMA, KNOWN_GATE_V25E_MANIFEST_SCHEMA, KNOWN_GATE_V26_EXPERIMENT_SCHEMA, KNOWN_GATE_V26_MANIFEST_SCHEMA, KNOWN_GATE_V26_READINESS_SCHEMA, UNKNOWN_LABEL, KnownGateClassifier, KnownGateConfig, build_known_gate_dataloader, build_known_gate_dataset, build_known_gate_model_for_config, evaluate_known_gate, known_gate_config_identity_sha256, known_gate_device, load_known_gate_config, seed_everything, sha256_file
from radar_ln3.known_gate_features import BASELINE_VARIANT, COURSE_PREFIX_FEATURE_SCHEMA, V21A_INTERFRAME_VARIANT, V21B_VARIANT, V21C_VARIANT, V21D_PREFIX_VARIANT, V21E_COURSE_PREFIX_VARIANT, V22A_RD_TEMPORAL_RESIDUAL_VARIANT, V24_COURSE_REASON_AWARE_BASELINE_VARIANT, V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT, V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT, V25A_GRADIENT_TELEMETRY_VARIANT, V25D_PRESENT_CLASS_MEAN_VARIANT, V25E_SHARED_GRADIENT_PROJECTION_VARIANT, V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.known_gate_gradient_decomposition import module_parameter_indices
from radar_ln3.known_gate_gradient_projection import V25E_PROJECTED_MODULE_NAMES, apply_historical_mixed_batch_projection


READINESS_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V21-known-gate-training-readiness.json"
)
MULTISEED_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json"
)
V26_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-"
    "readiness.json"
)




def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parent_order_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _capture_training_state(
    generator: torch.Generator,
    *,
    run_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator_state": generator.get_state(),
        "run_state": dict(run_state),
    }


def _restore_training_state(
    payload: Mapping[str, Any],
    generator: torch.Generator,
) -> dict[str, Any]:
    state = payload.get("training_state")
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_generator_state",
        "run_state",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("V21 checkpoint lacks complete epoch-boundary recovery state")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(state["torch_cuda_rng_states"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("V21 checkpoint CUDA RNG state count differs from this host")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, device="cpu") for value in cuda_states])
    generator.set_state(torch.as_tensor(state["dataloader_generator_state"], device="cpu"))
    run_state = state["run_state"]
    if not isinstance(run_state, Mapping):
        raise ValueError("V21 checkpoint training_state lacks run_state")
    return dict(run_state)


def _verify_contract(
    root: Path,
    config: KnownGateConfig,
) -> dict[str, Any]:
    paths = {
        "index": _resolve(root, config.data.index_path),
        "contract": _resolve(root, config.data.contract_path),
        "stats": _resolve(root, config.data.normalization_stats_path),
    }
    expected = {
        "index": config.data.expected_index_sha256,
        "contract": config.data.expected_contract_sha256,
        "stats": config.data.expected_stats_sha256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise ValueError(f"V21 {name} path/hash differs from the config: {path}")
    stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
    if (
        stats.get("schema_version") != "RADAR-LN3-V21"
        or stats.get("source_split") != "train"
        or int(stats.get("parent_count", -1)) != 986
        or stats.get("allowed_labels") != [
            "BalloonTarget",
            "BirdTarget",
            "ClutterTarget",
            "DroneTarget",
        ]
        or stats.get("forbidden_labels") != [UNKNOWN_LABEL]
        or int(dict(stats.get("excluded_label_parent_counts") or {}).get(UNKNOWN_LABEL, -1)) != 69
        or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
    ):
        raise ValueError("V21 normalization does not prove known-only train population")
    amplitude = dict(stats.get("amplitude") or {})
    if (
        abs(float(amplitude.get("center")) - config.data.rd_amplitude_center) > 1.0e-6
        or abs(float(amplitude.get("scale")) - config.data.rd_amplitude_scale) > 1.0e-6
    ):
        raise ValueError("V21 config RD normalization values differ from the stats")
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    strict = dict(contract.get("strict_open_set_contract") or {})
    task = dict(contract.get("task_contract") or {})
    normalization = dict(contract.get("normalization_contract") or {})
    scope = dict(contract.get("data_scope") or {})
    if (
        contract.get("schema_version") != KNOWN_GATE_CONTRACT_SCHEMA
        or strict.get("l1_task_name") != "known_gate"
        or strict.get("unknown_label") != UNKNOWN_LABEL
        or any(
            strict.get(field) is not False
            for field in (
                "unknown_is_model_class",
                "unknown_is_training_target",
                "unknown_is_normalization_population",
                "unknown_is_sampler_population",
                "unknown_is_optimizer_population",
                "unknown_is_checkpoint_ranking_population",
                "unknown_is_threshold_fitting_population",
            )
        )
        or int(strict.get("unknown_training_exposure", -1)) != 0
        or strict.get("cu_specialist_exists_in_v21") is not False
        or task.get("class_names") != list(KNOWN_GATE_CLASS_NAMES)
        or task.get("label_to_id") != dict(config.label_to_id)
        or int(task.get("train_parent_count", -1)) != 986
        or int(task.get("validation_parent_count", -1)) != 232
        or int(task.get("validation_clutter_parent_count", -1)) != 9
        or normalization.get("sha256") != expected["stats"]
        or normalization.get("fit_population") != "known_only"
        or int(scope.get("test_raw_payload_access_count", -1)) != 0
        or int(scope.get("test_rd_payload_access_count", -1)) != 0
    ):
        raise ValueError("V21 data contract violates the frozen KnownGate/open-set semantics")
    cache_evidence: dict[str, Any] | None = None
    if config.variant != BASELINE_VARIANT:
        cache_paths = {
            "manifest": _resolve(root, config.data.feature_cache_manifest_path or ""),
            "train": _resolve(root, config.data.train_feature_cache_path or ""),
            "validation": _resolve(root, config.data.validation_feature_cache_path or ""),
        }
        cache_expected = {
            "manifest": config.data.expected_feature_cache_manifest_sha256,
            "train": config.data.expected_train_feature_cache_sha256,
            "validation": config.data.expected_validation_feature_cache_sha256,
        }
        for name, path in cache_paths.items():
            if not path.is_file() or sha256_file(path) != cache_expected[name]:
                raise ValueError(f"V21 {config.variant} {name} cache path/hash differs")
        manifest = json.loads(cache_paths["manifest"].read_text(encoding="utf-8"))
        if  config.variant in {
            V21E_COURSE_PREFIX_VARIANT,
            V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
            V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
            V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
            V25A_GRADIENT_TELEMETRY_VARIANT,
            V25D_PRESENT_CLASS_MEAN_VARIANT,
            V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
            V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
        }:
            experiment_path = _resolve(
                root, config.data.feature_experiment_contract_path or ""
            )
            experiment_expected = (
                config.data.expected_feature_experiment_contract_sha256
            )
            experiment = (
                json.loads(experiment_path.read_text(encoding="utf-8"))
                if experiment_path.is_file()
                else {}
            )
            experiment_document_path = _resolve(
                root, str(experiment.get("experiment_document", ""))
            )
            manifest_caches = dict(manifest.get("cache_files") or {})
            manifest_train = dict(manifest_caches.get("train") or {})
            manifest_validation = dict(manifest_caches.get("val") or {})
            is_v24 = config.variant == V24_COURSE_REASON_AWARE_BASELINE_VARIANT
            is_v24a = config.variant == V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
            is_v24c = (
                config.variant == V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
            )
            is_v25a = config.variant == V25A_GRADIENT_TELEMETRY_VARIANT
            is_v25d = config.variant == V25D_PRESENT_CLASS_MEAN_VARIANT
            is_v25e = config.variant == V25E_SHARED_GRADIENT_PROJECTION_VARIANT
            is_v26 = config.variant == V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT
            audit_cache_files = dict(manifest.get("audit_cache_files") or {})
            audit_validation = dict(audit_cache_files.get("val_mixed") or {})
            audit_contract = dict(experiment.get("open_set_audit_contract") or {})
            audit_validation_path = _resolve(
                root, str(audit_validation.get("path", ""))
            )
            expected_manifest_schema = (
                KNOWN_GATE_V24A_MANIFEST_SCHEMA
                if is_v24a
                else (
                    KNOWN_GATE_V24C_MANIFEST_SCHEMA
                    if is_v24c
                    else (
                        KNOWN_GATE_V25A_MANIFEST_SCHEMA
                        if is_v25a
                        else (
                            KNOWN_GATE_V25D_MANIFEST_SCHEMA
                            if is_v25d
                            else (
                                KNOWN_GATE_V25E_MANIFEST_SCHEMA
                                if is_v25e
                                else (
                                KNOWN_GATE_V24_MANIFEST_SCHEMA
                                if is_v24
                                else KNOWN_GATE_V21E_MANIFEST_SCHEMA
                                )
                            )
                        )
                    )
                )
            )
            expected_feature_schema = (
                "cq08-current-event-course-plus-rd-peak-drift-mean-v1"
                if is_v24a
                else COURSE_PREFIX_FEATURE_SCHEMA
            )
            expected_experiment_schema = (
                KNOWN_GATE_V24A_EXPERIMENT_SCHEMA
                if is_v24a
                else (
                    KNOWN_GATE_V24C_EXPERIMENT_SCHEMA
                    if is_v24c
                    else (
                        KNOWN_GATE_V25A_EXPERIMENT_SCHEMA
                        if is_v25a
                        else (
                            KNOWN_GATE_V25D_EXPERIMENT_SCHEMA
                            if is_v25d
                            else (
                                KNOWN_GATE_V25E_EXPERIMENT_SCHEMA
                                if is_v25e
                                else (
                                KNOWN_GATE_V24_EXPERIMENT_SCHEMA
                                if is_v24
                                else (
                                    "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
                                    "experiment-contract-v1"
                                )
                                )
                            )
                        )
                    )
                )
            )
            expected_variant = (
                V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
                if is_v24a
                else (
                    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
                    if is_v24c
                    else (
                        V25A_GRADIENT_TELEMETRY_VARIANT
                        if is_v25a
                        else (
                            V25D_PRESENT_CLASS_MEAN_VARIANT
                            if is_v25d
                            else (
                                V25E_SHARED_GRADIENT_PROJECTION_VARIANT
                                if is_v25e
                                else (
                                V24_COURSE_REASON_AWARE_BASELINE_VARIANT
                                if is_v24
                                else V21E_COURSE_PREFIX_VARIANT
                                )
                            )
                        )
                    )
                )
            )
            if is_v26:
                expected_manifest_schema = KNOWN_GATE_V26_MANIFEST_SCHEMA
                expected_experiment_schema = KNOWN_GATE_V26_EXPERIMENT_SCHEMA
                expected_variant = V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT
            expected_not_executed = (
                "not_executed"
                if (is_v24 or is_v24a or is_v24c or is_v25a or is_v25d or is_v25e)
                else "未执行"
            )
            if is_v26:
                expected_not_executed = "not_executed"
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or manifest.get("schema_version") != expected_manifest_schema
                or manifest.get("feature_schema") != expected_feature_schema
                or manifest.get("feature_names") != list(config.feature_names)
                or manifest.get("core_data_contract_sha256") != expected["contract"]
                or manifest.get("experiment_contract_sha256") != experiment_expected
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
                or manifest.get("sample_id_is_model_input") is not False
                or manifest.get("unknown_excluded") is not True
                or manifest.get("validation_label_values_used_for_readiness_gate")
                is not False
                or _resolve(root, str(manifest_train.get("path", "")))
                != cache_paths["train"]
                or manifest_train.get("sha256") != cache_expected["train"]
                or _resolve(root, str(manifest_validation.get("path", "")))
                != cache_paths["validation"]
                or manifest_validation.get("sha256") != cache_expected["validation"]
                or experiment.get("schema_version") != expected_experiment_schema
                or experiment.get("internal_variant") != expected_variant
                or experiment.get("feature_names") != list(config.feature_names)
                or dict(experiment.get("evaluation") or {}).get(
                    "primary_hypothesis_labels"
                )
                != [
                    "DroneTarget",
                    "BirdTarget",
                    "BalloonTarget",
                    "ClutterTarget",
                ]
                or dict(experiment.get("evaluation") or {}).get("safety_labels")
                != []
                or experiment.get("core_data_contract_sha256") != expected["contract"]
                or not experiment_document_path.is_file()
                or sha256_file(experiment_document_path)
                != experiment.get("experiment_document_sha256")
                or (
                    not is_v24
                    and not is_v24a
                    and not is_v24c
                    and not is_v25a
                    and not is_v25d
                    and not is_v25e
                    and not is_v26
                    and experiment.get("post_hoc_feature_discovery") is not True
                )
                or (
                    (
                        is_v24
                        or is_v24a
                        or is_v24c
                        or is_v25a
                        or is_v25d
                        or is_v25e
                        or is_v26
                    )
                    and experiment.get("fresh_training_not_finetune") is not True
                )
                or experiment.get("test") != expected_not_executed
                or int(experiment.get("test_payload_access_count", -1)) != 0
                or (
                    is_v26
                    and (
                        not audit_validation_path.is_file()
                        or audit_validation.get("sha256")
                        != sha256_file(audit_validation_path)
                        or audit_validation.get("split") != "val"
                        or int(audit_validation.get("event_count", -1)) != 4725
                        or int(audit_validation.get("parent_count", -1)) != 247
                        or int(audit_validation.get("unknown_parent_count", -1)) != 15
                        or int(audit_validation.get("feature_width", -1)) != 1
                        or audit_contract.get("validation_mixed_cache")
                        != audit_validation.get("path")
                        or audit_contract.get("validation_mixed_cache_sha256")
                        != audit_validation.get("sha256")
                        or audit_contract.get("test_mixed_cache_sha256")
                        != "materialized_after_checkpoint_frozen"
                    )
                )
                or (
                    is_v26
                    and (
                        not dict(experiment.get("open_set_audit_contract") or {}).get(
                            "audit_only", False
                        )
                        or dict(experiment.get("open_set_audit_contract") or {}).get(
                            "unknown_is_model_class"
                        )
                        is not False
                        or dict(experiment.get("open_set_audit_contract") or {}).get(
                            "unknown_is_training_target"
                        )
                        is not False
                        or dict(experiment.get("open_set_audit_contract") or {}).get(
                            "test_access_stage"
                        )
                        != "after_known_only_checkpoint_frozen"
                    )
                )
            ):
                raise ValueError(
                    "KnownGate course feature manifest/experiment contract is stale"
                )
        elif (
            manifest.get("schema_version") != KNOWN_GATE_MANIFEST_SCHEMA
            or manifest.get("data_contract_sha256") != expected["contract"]
            or manifest.get("test_excluded") is not True
            or int(manifest.get("test_payload_access_count", -1)) != 0
            or manifest.get("valid_mask_is_model_input") is not False
        ):
            raise ValueError("V21 feature manifest violates cache/open-set semantics")
        cache_evidence = {
            "paths": {name: str(path) for name, path in cache_paths.items()},
            "hashes": {name: str(cache_expected[name]) for name in cache_expected},
            **(
                {}
                if config.variant
                not in {
                    V21A_INTERFRAME_VARIANT,
                    V21B_VARIANT,
                    V21C_VARIANT,
                    V21D_PREFIX_VARIANT,
                    V21E_COURSE_PREFIX_VARIANT,
                    V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
                    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
                    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
                    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
                    V25A_GRADIENT_TELEMETRY_VARIANT,
                    V25D_PRESENT_CLASS_MEAN_VARIANT,
                    V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
                    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
                }
                else {
                    "experiment_contract_path": str(experiment_path),
                    "experiment_contract_sha256": str(experiment_expected),
                }
            ),
        }
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": {
            "index_sha256": expected["index"],
            "contract_sha256": expected["contract"],
            "normalization_sha256": expected["stats"],
            **(
                {}
                if cache_evidence is None
                else {
                    "feature_manifest_sha256": cache_evidence["hashes"]["manifest"],
                    "train_feature_cache_sha256": cache_evidence["hashes"]["train"],
                    "validation_feature_cache_sha256": cache_evidence["hashes"]["validation"],
                    **(
                        {}
                        if config.variant
                        not in {
                            V21A_INTERFRAME_VARIANT,
                            V21B_VARIANT,
                            V21C_VARIANT,
                            V21D_PREFIX_VARIANT,
                            V21E_COURSE_PREFIX_VARIANT,
                            V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
                            V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
                            V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
                            V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
                            V25A_GRADIENT_TELEMETRY_VARIANT,
                            V25D_PRESENT_CLASS_MEAN_VARIANT,
                            V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
                            V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
                        }
                        else {
                            "feature_experiment_contract_sha256": cache_evidence[
                                "experiment_contract_sha256"
                            ]
                        }
                    ),
                }
            ),
        },
        "feature_cache": cache_evidence,
    }


def _verify_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if config.variant == V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT:
        path = (root / V26_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V26 formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        expected_hashes = dict(report.get("contract_hashes") or {})
        actual_hashes = dict(contract_evidence.get("hashes") or {})
        required_hashes = {
            "index_sha256": actual_hashes.get("index_sha256"),
            "core_contract_sha256": actual_hashes.get("contract_sha256"),
            "normalization_sha256": actual_hashes.get("normalization_sha256"),
            "experiment_contract_sha256": actual_hashes.get(
                "feature_experiment_contract_sha256"
            ),
            "feature_manifest_sha256": actual_hashes.get("feature_manifest_sha256"),
            "train_feature_cache_sha256": actual_hashes.get(
                "train_feature_cache_sha256"
            ),
            "validation_feature_cache_sha256": actual_hashes.get(
                "validation_feature_cache_sha256"
            ),
        }
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V26_READINESS_SCHEMA
            or report.get("variant") != V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT
            or not checks
            or not all(value is True for value in checks.values())
            or not isinstance(report.get("initial_model_state_sha256"), str)
            or len(str(report.get("initial_model_state_sha256", ""))) != 64
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(expected_hashes.get(key) != value for key, value in required_hashes.items())
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("open_set_audit") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V26 readiness is missing, stale, failed or outside the Known-only contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V26_READINESS_SCHEMA,
            "status": "passed",
            # Training uses this sealed value to prove that the freshly
            # constructed treatment model is byte-identical to the model
            # audited during prepare.  Do not drop it when returning the
            # validated readiness view.
            "initial_model_state_sha256": str(
                report.get("initial_model_state_sha256", "")
            ),
            "paired_initialization": dict(
                report.get("paired_initialization") or {}
            ),
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == BASELINE_VARIANT and config.train.seed != 42:
        return _verify_baseline_multiseed_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
    path = (root / READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21 formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    binding = dict(dict(report.get("configs") or {}).get(config.variant) or {})
    checks = dict(report.get("checks") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
    }
    if config.variant != BASELINE_VARIANT:
        actual_required.update(
            {
                "feature_manifest_sha256": contract_evidence["hashes"]["feature_manifest_sha256"],
                "train_feature_cache_sha256": contract_evidence["hashes"]["train_feature_cache_sha256"],
                "validation_feature_cache_sha256": contract_evidence["hashes"]["validation_feature_cache_sha256"],
            }
        )
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError("V21 readiness is missing, stale, failed or outside the sealed contract")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_READINESS_SCHEMA,
        "status": "passed",
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }














def _verify_baseline_multiseed_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if config.train.seed not in KNOWN_GATE_BASELINE_MULTISEED_SEEDS[1:]:
        raise PermissionError("V21 additional Baseline seed is outside the frozen seed set")
    path = (root / MULTISEED_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21 Baseline multiseed training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(dict(report.get("configs") or {}).get(str(config.train.seed)) or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
    }
    base_readiness_path = (root / READINESS_RELATIVE).resolve()
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA
        or report.get("seeds") != list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS)
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or not base_readiness_path.is_file()
        or report.get("base_readiness_sha256") != sha256_file(base_readiness_path)
        or report.get("new_seed_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21 Baseline multiseed readiness is missing, stale, failed or outside the sealed contract"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
        "status": "passed",
        "seed": config.train.seed,
        "validation_used_for_cross_seed_development_selection": True,
        "test": "未执行",
    }


def _selection(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(evaluation.get("parent_metrics") or {})
    rows = list(metrics.get("per_class") or [])
    if len(rows) != 2 or any(int(row.get("support", 0)) <= 0 for row in rows):
        raise ValueError("V21 checkpoint selection requires two non-zero-support classes")
    weak = min(
        float(row[field])
        for row in rows
        for field in ("precision", "recall")
    )
    macro_f1 = float(metrics["macro_f1"])
    if not np.isfinite(weak) or not np.isfinite(macro_f1):
        raise ValueError("V21 checkpoint selection metric is non-finite")
    return {
        "metric": KNOWN_GATE_SELECTION_METRIC,
        "primary_metric": "parent_min_class_precision_recall",
        "primary_value": weak,
        "tie_break_metric": "parent_macro_f1",
        "tie_break_value": macro_f1,
        "ranking_key": [weak, macro_f1],
    }


def _is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return tuple(float(value) for value in candidate["ranking_key"]) > tuple(
        float(value) for value in incumbent["ranking_key"]
    )


def _checkpoint_payload(
    model: KnownGateClassifier,
    optimizer: torch.optim.Optimizer,
    *,
    config: KnownGateConfig,
    config_path: Path,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any],
    readiness_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": config.train.checkpoint_schema,
        "task_name": "known_gate",
        "variant": config.variant,
        "config_name": config.config_name,
        "config_identity_sha256": known_gate_config_identity_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "readiness_sha256": readiness_sha256,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "training_state": dict(training_state),
        "full_parameter_training": True,
        "unknown_is_model_class": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    model: KnownGateClassifier,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: KnownGateConfig,
    config_path: Path,
    readiness_sha256: str,
    map_location: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if (
        payload.get("checkpoint_schema") != config.train.checkpoint_schema
        or payload.get("task_name") != "known_gate"
        or payload.get("variant") != config.variant
        or payload.get("config_identity_sha256") != known_gate_config_identity_sha256(config)
        or payload.get("config_file_sha256") != sha256_file(config_path)
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("full_parameter_training") is not True
        or payload.get("unknown_is_model_class") is not False
    ):
        raise ValueError("V21 checkpoint schema/config/readiness binding differs")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return dict(payload)


def _model_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tr_features": tensors["tr_features"],
        "tr_missing_mask": tensors["tr_missing_mask"],
        "tr_dt_seconds": tensors["tr_dt_seconds"],
        "tr_dt_valid": tensors["tr_dt_valid"],
        "tr_dt_reason": tensors["tr_dt_reason"],
        "rd_frames": tensors["rd_frames"],
        "rd_frame_mask": tensors["rd_frame_mask"],
        "rd_quality": tensors["rd_quality"],
        "tr_padding_mask": tensors["tr_padding_mask"],
        "known_gate_features": tensors.get("known_gate_features"),
        "known_gate_feature_valid": tensors.get("known_gate_feature_valid"),
    }


def _performance(evaluation: Mapping[str, Any], target: float) -> dict[str, Any]:
    rows = list(dict(evaluation.get("parent_metrics") or {}).get("per_class") or [])
    guardrails: list[dict[str, Any]] = []
    values: list[float] = []
    for row in rows:
        metrics = {field: float(row[field]) for field in ("precision", "recall", "f1")}
        values.extend(metrics.values())
        guardrails.append(
            {
                "class": str(row["class"]),
                "support": int(row["support"]),
                **metrics,
                "target": float(target),
                "passed": all(value >= target for value in metrics.values()),
            }
        )
    passed = bool(guardrails) and all(row["passed"] for row in guardrails)
    return {
        "performance_status": "passed" if passed else "failed",
        "target_min_per_class_precision_recall_f1": float(target),
        "observed_min_per_class_precision_recall_f1": min(values) if values else None,
        "guardrails": guardrails,
    }


def train_known_gate(
    project_root: Path,
    config_path: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = config_path.resolve()
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    artifact_dir = _resolve(root, config.train.formal_artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            "refusing to overwrite a non-empty V21 formal artifact; use --resume-checkpoint "
            f"at an epoch boundary or archive it: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(artifact_dir / "training.log")
    started = time.perf_counter()
    try:
        seed_everything(config.train.seed)
        train_dataset = build_known_gate_dataset(root, config, split_name="train")
        validation_dataset = build_known_gate_dataset(root, config, split_name="val")
        if train_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 953,
            KNOWN_GATE_CLASS_NAMES[1]: 33,
        } or validation_dataset.label_counts() != {
            KNOWN_GATE_CLASS_NAMES[0]: 223,
            KNOWN_GATE_CLASS_NAMES[1]: 9,
        }:
            raise ValueError("V21 KnownGate train/validation parent support changed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.train.dataloader_seed)
        train_loader = build_known_gate_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
        )
        model, baseline_reference, paired_initialization = build_known_gate_model_for_config(root, config)
        initial_state_sha256 = _state_dict_sha256(model.state_dict())
        if baseline_reference is not None:
            if config.variant == V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT:
                baseline_initial_state = _state_dict_sha256(
                    baseline_reference.state_dict()
                )
                if (
                    initial_state_sha256
                    != str(readiness.get("initial_model_state_sha256", ""))
                    or paired_initialization.get("paired_baseline") is not True
                    or paired_initialization.get(
                        "shared_parameters_and_buffers_exactly_equal"
                    )
                    is not True
                    or paired_initialization.get("new_fusion_columns_exactly_zero")
                    is not True
                    or int(paired_initialization.get("new_fusion_columns", -1)) != 1
                    or not baseline_initial_state
                ):
                    raise AssertionError(
                        "V26 fresh Known-only initialization differs from its sealed readiness"
                    )
                paired_initialization["sealed_v26_initial_model_state_reproduced"] = True
                paired_initialization["sealed_v26_initial_model_state_sha256"] = (
                    initial_state_sha256
                )
            audit_loader = build_known_gate_dataloader(
                train_dataset,
                batch_size=min(config.train.batch_size, len(train_dataset)),
                shuffle=False,
            )
            probe = next(iter(audit_loader))
            same_second_probe: Mapping[str, Any] | None = None
            model.eval()
            baseline_reference.eval()
            with torch.inference_mode():
                treatment_output = model(**_model_inputs(probe))
                treatment_logits = treatment_output["sequence_logits"]
                baseline_inputs = _model_inputs(probe)
                baseline_evidence_dim = baseline_reference.config.model.evidence_dim
                if baseline_evidence_dim:
                    baseline_inputs["known_gate_features"] = baseline_inputs[
                        "known_gate_features"
                    ][..., :baseline_evidence_dim]
                    baseline_inputs["known_gate_feature_valid"] = baseline_inputs[
                        "known_gate_feature_valid"
                    ][..., :baseline_evidence_dim]
                else:
                    baseline_inputs["known_gate_features"] = None
                    baseline_inputs["known_gate_feature_valid"] = None
                baseline_logits = baseline_reference(**baseline_inputs)["sequence_logits"]
            if not torch.equal(treatment_logits, baseline_logits):
                raise AssertionError("V21 Treatment initial logits differ from the paired Baseline")
            paired_initialization["initial_logits_exactly_equal"] = True
            paired_initialization["initial_logits_sha256"] = hashlib.sha256(
                treatment_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
            ).hexdigest()
            del baseline_reference, audit_loader, probe
            del treatment_output, treatment_logits, baseline_logits
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V21 must train every model parameter")
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        v25e_module_indices = (
            module_parameter_indices(model, tuple(trainable))
            if config.variant == V25E_SHARED_GRADIENT_PROJECTION_VARIANT
            else None
        )
        device = known_gate_device(config)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_selection: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        resumed_from: str | None = None
        if resume_checkpoint is not None:
            resume_path = resume_checkpoint.resolve()
            payload = _load_checkpoint(
                resume_path,
                model,
                optimizer,
                config=config,
                config_path=config_path,
                readiness_sha256=readiness["sha256"],
                map_location=device,
            )
            run_state = _restore_training_state(payload, generator)
            start_epoch = int(run_state["next_epoch"])
            global_step = int(run_state["global_step"])
            best_epoch = int(run_state["best_epoch"])
            best_selection = run_state.get("best_selection")
            history = list(run_state.get("history") or [])
            if start_epoch != int(payload["epoch"]) + 1 or len(history) != int(payload["epoch"]):
                raise ValueError("V21 resume state is not an epoch boundary")
            if not 1 <= start_epoch <= config.train.epochs + 1:
                raise ValueError("V21 resume next_epoch is outside the fixed 20-epoch run")
            resumed_from = str(resume_path)
        best_checkpoint = _resolve(root, config.train.checkpoint_path)
        last_checkpoint = _resolve(root, config.train.last_checkpoint_path)
        steps_per_epoch = len(train_loader)
        planned_steps = max(0, config.train.epochs - start_epoch + 1) * steps_per_epoch
        logger.info(
            "run=start | task=known_gate | variant=%s | device=%s | train_parents=%d | val_parents=%d | epochs=20 | planned_steps=%d | full_parameter_training=true | unknown_training_exposure=0",
            config.variant,
            device,
            len(train_dataset),
            len(validation_dataset),
            planned_steps,
        )
        def _new_loss_mass_epoch() -> dict[str, dict[str, Any]]:
            return {
                scope: {
                    "contributing_batch_count": 0,
                    "pure_batch_count": 0,
                    "mixed_batch_count": 0,
                    "effective_class_mass_sum": [0.0, 0.0],
                    "mixed_batches_exactly_equal_class_mass": True,
                    "pure_batches_have_single_class_mass": True,
                }
                for scope in ("final", "prefix")
            }

        def _record_loss_mass(
            epoch_mass: dict[str, dict[str, Any]],
            *,
            scope: str,
            mass: torch.Tensor,
            contributing: bool,
        ) -> None:
            if config.variant != V25D_PRESENT_CLASS_MEAN_VARIANT or not contributing:
                return
            values = [float(value) for value in mass.detach().cpu().tolist()]
            if len(values) != 2 or any(not np.isfinite(value) or value < 0.0 for value in values):
                raise AssertionError("V25d loss mass is malformed")
            target = epoch_mass[scope]
            target["contributing_batch_count"] += 1
            target["effective_class_mass_sum"] = [
                float(before + observed)
                for before, observed in zip(
                    target["effective_class_mass_sum"], values, strict=True
                )
            ]
            present = sum(value > 0.0 for value in values)
            if present == 2:
                target["mixed_batch_count"] += 1
                target["mixed_batches_exactly_equal_class_mass"] = bool(
                    target["mixed_batches_exactly_equal_class_mass"]
                    and values == [0.5, 0.5]
                )
            elif present == 1:
                target["pure_batch_count"] += 1
                target["pure_batches_have_single_class_mass"] = bool(
                    target["pure_batches_have_single_class_mass"]
                    and values in ([1.0, 0.0], [0.0, 1.0])
                )
            else:
                raise AssertionError("V25d contributing loss has no present class mass")

        def _new_v25e_epoch_telemetry() -> dict[str, Any]:
            return {
                "batch_count": 0,
                "mixed_known_clutter_batch_count": 0,
                "pure_batch_count": 0,
                "projection_batch_count": 0,
                "projected_module_count": 0,
                "negative_conflict_count": {
                    module: 0 for module in V25E_PROJECTED_MODULE_NAMES
                },
                "projection_count": {
                    module: 0 for module in V25E_PROJECTED_MODULE_NAMES
                },
                "max_historical_loss_reconstruction_absolute_error": 0.0,
                "rd_and_classifier_gradients_exactly_unchanged": True,
                "pure_batches_used_historical_backward": True,
            }

        for epoch in range(start_epoch, config.train.epochs + 1):
            model.train()
            total_loss = 0.0
            final_loss = 0.0
            prefix_loss = 0.0
            gradient_max = 0.0
            epoch_loss_mass = _new_loss_mass_epoch()
            epoch_v25e_telemetry = _new_v25e_epoch_telemetry()
            epoch_order: list[str] = []
            epoch_started = time.perf_counter()
            for step, batch in enumerate(train_loader, start=1):
                epoch_order.extend(str(value) for value in batch["parent_sample_ids"])
                tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(**_model_inputs(tensors))
                objective = parent_final_task_ce_with_prefix_auxiliary(
                    output["sequence_logits"],
                    tensors["labels"],
                    tensors["tr_padding_mask"],
                    class_weights=class_weights,
                    prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                    prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                    class_weight_normalization=config.train.class_weight_normalization,
                )
                loss = objective["loss"]
                is_v25e_mixed = bool(
                    config.variant == V25E_SHARED_GRADIENT_PROJECTION_VARIANT
                    and tensors["labels"].shape == (2,)
                    and sorted(int(value) for value in tensors["labels"].detach().cpu().tolist())
                    == [0, 1]
                )
                _record_loss_mass(
                    epoch_loss_mass,
                    scope="final",
                    mass=objective["final_effective_class_mass"],
                    contributing=True,
                )
                _record_loss_mass(
                    epoch_loss_mass,
                    scope="prefix",
                    mass=objective["prefix_effective_class_mass"],
                    contributing=bool(
                        int(objective["eligible_prefix_parent_count"].item()) > 0
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=is_v25e_mixed)
                v25e_step_audit: dict[str, Any] | None = None
                if is_v25e_mixed:
                    if v25e_module_indices is None:
                        raise AssertionError("V25e mixed batch has no module partition")
                    v25e_step_audit = apply_historical_mixed_batch_projection(
                        parameters=tuple(trainable),
                        module_indices=v25e_module_indices,
                        sequence_logits=output["sequence_logits"],
                        labels=tensors["labels"],
                        padding_mask=tensors["tr_padding_mask"],
                        class_weights=class_weights,
                        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
                        original_loss=loss,
                    )
                    epoch_v25e_telemetry["projection_batch_count"] += 1
                    epoch_v25e_telemetry["projected_module_count"] += int(
                        v25e_step_audit["projected_module_count"]
                    )
                    epoch_v25e_telemetry[
                        "max_historical_loss_reconstruction_absolute_error"
                    ] = max(
                        float(
                            epoch_v25e_telemetry[
                                "max_historical_loss_reconstruction_absolute_error"
                            ]
                        ),
                        float(
                            v25e_step_audit[
                                "historical_loss_reconstruction_absolute_error"
                            ]
                        ),
                    )
                    epoch_v25e_telemetry[
                        "rd_and_classifier_gradients_exactly_unchanged"
                    ] = bool(
                        epoch_v25e_telemetry[
                            "rd_and_classifier_gradients_exactly_unchanged"
                        ]
                        and v25e_step_audit[
                            "rd_and_classifier_gradients_exactly_unchanged"
                        ]
                    )
                    for module in V25E_PROJECTED_MODULE_NAMES:
                        module_audit = dict(
                            dict(v25e_step_audit["shared_module_projection"])[module]
                        )
                        if module_audit.get("pre_cosine") is not None and float(
                            module_audit["pre_cosine"]
                        ) < 0.0:
                            epoch_v25e_telemetry["negative_conflict_count"][module] += 1
                        if module_audit.get("projected") is True:
                            epoch_v25e_telemetry["projection_count"][module] += 1
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item()
                )
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("V21 training produced non-finite loss or gradient")
                loss_value = float(loss.item())
                final_value = float(objective["final_state_loss"].item())
                prefix_value = float(objective["prefix_auxiliary_loss"].item())
                optimizer.step()
                global_step += 1
                total_loss += loss_value
                final_loss += final_value
                prefix_loss += prefix_value
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=%s | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        config.variant,
                        epoch,
                        step,
                        steps_per_epoch,
                        loss_value,
                        final_value,
                        prefix_value,
                        grad_norm,
                    )
                # V25e temporarily retains the forward graph to read the two
                # historical parent gradients.  The final autograd.grad call
                # releases that graph; drop the remaining Python references at
                # the batch boundary so a long run cannot accumulate CUDA
                # tensors through loop locals or telemetry intermediates.
            if len(epoch_order) != len(train_dataset) or len(set(epoch_order)) != len(epoch_order):
                raise ValueError("V21 full epoch did not consume each train parent exactly once")
            epoch_order_sha256 = _parent_order_sha256(epoch_order)
            validation = evaluate_known_gate(
                model,
                validation_dataset,
                config=config,
                device=device,
            )
            selection = _selection(validation)
            selected_as_best = _is_better(selection, best_selection)
            if selected_as_best:
                best_epoch = epoch
                best_selection = selection
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": total_loss / steps_per_epoch,
                "train_final_state_loss": final_loss / steps_per_epoch,
                "train_prefix_auxiliary_loss": prefix_loss / steps_per_epoch,
                "gradient_norm_max": gradient_max,
                "selection_metric": KNOWN_GATE_SELECTION_METRIC,
                "checkpoint_selection": selection,
                "validation": {key: value for key, value in validation.items() if key != "records"},
                "parent_order_sha256": epoch_order_sha256,
                "parent_order_count": len(epoch_order),
                "parent_order_unique_count": len(set(epoch_order)),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                **(
                    {}
                    if config.variant != V25D_PRESENT_CLASS_MEAN_VARIANT
                    else {"loss_class_mass": epoch_loss_mass}
                ),
                **(
                    {}
                    if config.variant != V25E_SHARED_GRADIENT_PROJECTION_VARIANT
                    else {
                        "shared_gradient_projection": epoch_v25e_telemetry,
                    }
                ),
            }
            history.append(epoch_record)
            training_state = _capture_training_state(
                generator,
                run_state={
                    "next_epoch": epoch + 1,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_selection": best_selection,
                    "history": history,
                },
            )
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                config=config,
                config_path=config_path,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_record,
                training_state=training_state,
                readiness_sha256=readiness["sha256"],
            )
            _save_checkpoint(last_checkpoint, checkpoint)
            epoch_path = _resolve(root, config.train.epoch_checkpoint_dir) / f"epoch-{epoch:03d}.pt"
            _save_checkpoint(epoch_path, checkpoint)
            if selected_as_best:
                _save_checkpoint(best_checkpoint, checkpoint)
            logger.info(
                "epoch | variant=%s | epoch=%02d/20 | loss=%.6f | selection=%s:%s | tie=%s:%s | best=%s | order_sha256=%s | elapsed=%s",
                config.variant,
                epoch,
                epoch_record["train_loss"],
                selection["primary_metric"],
                format_metric(selection["primary_value"]),
                selection["tie_break_metric"],
                format_metric(selection["tie_break_value"]),
                selected_as_best,
                epoch_record["parent_order_sha256"],
                format_duration(epoch_record["elapsed_seconds"]),
            )
        if not best_checkpoint.is_file() or best_epoch <= 0:
            raise ValueError("V21 training did not produce a best checkpoint")
        _load_checkpoint(
            best_checkpoint,
            model,
            None,
            config=config,
            config_path=config_path,
            readiness_sha256=readiness["sha256"],
            map_location=device,
        )
        best_validation = evaluate_known_gate(
            model,
            validation_dataset,
            config=config,
            device=device,
        )
        sample_detail = {
            "schema_version": f"RADAR-LN3-V21-{config.variant}-known-gate-sample-detail-v1",
            "task_name": "known_gate",
            "variant": config.variant,
            "seed": config.train.seed,
            "dataloader_seed": config.train.dataloader_seed,
            "checkpoint": str(best_checkpoint),
            "epoch": best_epoch,
            "counts": {key: len(value) for key, value in best_validation["records"].items()},
            "records": best_validation["records"],
        }
        sample_detail_path = artifact_dir / "sample_detail_information.json"
        sample_detail_path.write_text(
            json.dumps(sample_detail, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        performance = _performance(best_validation, config.train.task_min_precision_recall_target)
        final_state_sha256 = _state_dict_sha256(model.state_dict())
        rd_temporal_residual_training_audit: dict[str, Any] | None = None
        present_class_mean_loss_mass_telemetry: dict[str, Any] | None = None
        shared_gradient_projection_telemetry: dict[str, Any] | None = None
        result = {
            "status": "completed",
            **performance,
            "task_name": "known_gate",
            "variant": config.variant,
            "seed": config.train.seed,
            "dataloader_seed": config.train.dataloader_seed,
            "config_name": config.config_name,
            "config_path": str(config_path),
            "device": str(device),
            "train_parent_count": len(train_dataset),
            "validation_parent_count": len(validation_dataset),
            "validation_clutter_support": 9,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_checkpoint_selection": best_selection,
            "best_validation_metrics": {key: value for key, value in best_validation.items() if key != "records"},
            "history": history,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "sample_detail_information": str(sample_detail_path),
            "trainable_parameter_count": trainable_parameter_count,
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "initial_model_state_sha256": initial_state_sha256,
            "best_model_state_sha256": final_state_sha256,
            "model_state_changed_from_initialization": initial_state_sha256 != final_state_sha256,
            "paired_initialization": paired_initialization,
            **(
                {}
                if rd_temporal_residual_training_audit is None
                else {
                    "rd_temporal_residual_training_audit": (
                        rd_temporal_residual_training_audit
                    )
                }
            ),
            **(
                {}
                if present_class_mean_loss_mass_telemetry is None
                else {
                    "present_class_mean_loss_mass_telemetry": (
                        present_class_mean_loss_mass_telemetry
                    )
                }
            ),
            **(
                {}
                if shared_gradient_projection_telemetry is None
                else {
                    "shared_gradient_projection_telemetry": (
                        shared_gradient_projection_telemetry
                    )
                }
            ),
            "training_readiness": readiness,
            "data_contract": contract,
            "dataloader_contract": {
                "seed": config.train.dataloader_seed,
                "independent_generator": True,
                "epoch_parent_order_hashes": [
                    {
                        "epoch": int(row["epoch"]),
                        "sha256": row["parent_order_sha256"],
                        "count": int(row["parent_order_count"]),
                    }
                    for row in history
                ],
                "cuda_deterministic_algorithms": False,
                "determinism_claim": "fixed_initialization_and_parent_order_not_bitwise_cuda_determinism",
            },
            "resume": {
                "resumed": resumed_from is not None,
                "checkpoint": resumed_from,
                "start_epoch": start_epoch,
                "epoch_boundary_only": True,
            },
            "unknown_is_model_class": False,
            "local_argmax_is_product_action": False,
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "formal_rejection": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
            "deployment_blockers": [
                "multi_seed_confirmation_not_executed",
                "parent_grouped_oof_thresholds_not_fitted",
                "full_cascade_not_executed",
                "test_not_executed",
                "formal_rejection_not_executed",
                "stable_not_executed",
                "p95_not_executed",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        (artifact_dir / "training-summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "run=complete | task=known_gate | variant=%s | status=completed | performance_status=%s | best_epoch=%02d | summary=%s",
            config.variant,
            result["performance_status"],
            best_epoch,
            artifact_dir / "training-summary.json",
        )
        return result
    except Exception as error:
        failure = {
            "status": "failed",
            "performance_status": "not_evaluated",
            "task_name": "known_gate",
            "variant": config.variant,
            "config_name": config.config_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "unknown_is_model_class": False,
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "full_cascade_evaluation": "not_executed",
            "formal_rejection": "not_executed",
            "stable_evaluation": "not_executed",
            "p95_latency": "not_executed",
            "deployment_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (artifact_dir / "training-summary.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.exception("run=failed | task=known_gate | variant=%s", config.variant)
        raise
    finally:
        close_training_logger(logger)



