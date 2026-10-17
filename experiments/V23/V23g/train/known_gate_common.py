from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA, KNOWN_GATE_BASELINE_MULTISEED_SEEDS, KNOWN_GATE_CLASS_NAMES, KNOWN_GATE_CONTRACT_SCHEMA, KNOWN_GATE_MANIFEST_SCHEMA, KNOWN_GATE_READINESS_SCHEMA, KNOWN_GATE_SELECTION_METRIC, KNOWN_GATE_V21A_MANIFEST_SCHEMA, KNOWN_GATE_V21A_READINESS_SCHEMA, KNOWN_GATE_V21B_READINESS_SCHEMA, KNOWN_GATE_V21C_READINESS_SCHEMA, KNOWN_GATE_V21D_MANIFEST_SCHEMA, KNOWN_GATE_V21D_READINESS_SCHEMA, KNOWN_GATE_V21E_MANIFEST_SCHEMA, KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA, KNOWN_GATE_V21E_MULTISEED_SEEDS, KNOWN_GATE_V21E_READINESS_SCHEMA, KNOWN_GATE_V22A_EXPERIMENT_SCHEMA, KNOWN_GATE_V22A_MANIFEST_SCHEMA, KNOWN_GATE_V22A_READINESS_SCHEMA, KNOWN_GATE_V24_EXPERIMENT_SCHEMA, KNOWN_GATE_V24_MANIFEST_SCHEMA, KNOWN_GATE_V24_READINESS_SCHEMA, KNOWN_GATE_V24A_EXPERIMENT_SCHEMA, KNOWN_GATE_V24A_MANIFEST_SCHEMA, KNOWN_GATE_V24A_READINESS_SCHEMA, KNOWN_GATE_V24C_EXPERIMENT_SCHEMA, KNOWN_GATE_V24C_MANIFEST_SCHEMA, KNOWN_GATE_V24C_READINESS_SCHEMA, KNOWN_GATE_V25A_EXPERIMENT_SCHEMA, KNOWN_GATE_V25A_MANIFEST_SCHEMA, KNOWN_GATE_V25A_READINESS_SCHEMA, KNOWN_GATE_V25D_EXPERIMENT_SCHEMA, KNOWN_GATE_V25D_MANIFEST_SCHEMA, KNOWN_GATE_V25D_READINESS_SCHEMA, KNOWN_GATE_V25E_EXPERIMENT_SCHEMA, KNOWN_GATE_V25E_MANIFEST_SCHEMA, KNOWN_GATE_V25E_READINESS_SCHEMA, KNOWN_GATE_V26_EXPERIMENT_SCHEMA, KNOWN_GATE_V26_MANIFEST_SCHEMA, KNOWN_GATE_V26_READINESS_SCHEMA, UNKNOWN_LABEL, KnownGateConfig, sha256_file
from radar_ln3.known_gate_features import (
    BASELINE_VARIANT,
    COURSE_PREFIX_FEATURE_SCHEMA,
    D_RANGE_PREFIX_FEATURE_SCHEMA,
    V21A_INTERFRAME_VARIANT,
    V21A_VARIANT,
    V21B_VARIANT,
    V21C_VARIANT,
    V21D_PREFIX_VARIANT,
    V21D_VARIANT,
    V21E_COURSE_PREFIX_VARIANT,
    V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
    V25A_GRADIENT_TELEMETRY_VARIANT,
    V25D_PRESENT_CLASS_MEAN_VARIANT,
    V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
)
from radar_ln3.known_gate_gradient_projection import V25E_PROJECTED_MODULE_NAMES, V25E_UNCHANGED_MODULE_NAMES


READINESS_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V21-known-gate-training-readiness.json"
)
MULTISEED_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json"
)
V21A_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21a-known-gate-interframe-spectral-training-readiness.json"
)
V21B_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21b-known-gate-modal-consistency-clutter-drone-training-readiness.json"
)
V21B_FORMAL_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21b-cq08-liquid-known-gate-modal-consistency-clutter-drone-"
    "seed43-config.json"
)
V21C_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-training-readiness.json"
)
V21C_FORMAL_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-clutter-drone-"
    "seed43-config.json"
)
V21D_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-training-readiness.json"
)
V21D_FORMAL_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21d-cq08-liquid-known-gate-causal-prefix-d-range-clutter-bird-"
    "seed43-config.json"
)
V21E_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-training-readiness.json"
)
V21E_FORMAL_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-config.json"
)
V21E_MULTISEED_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
    "multiseed-training-readiness.json"
)
V22A_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22a-known-gate-rd-temporal-residual-training-readiness.json"
)
V24_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness.json"
)
V24A_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-training-readiness.json"
)
V24C_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-training-readiness.json"
)
V25A_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-training-readiness.json"
)
V25D_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25d-known-gate-course-present-class-mean-training-readiness.json"
)
V25E_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25e-known-gate-course-shared-gradient-conflict-projection-"
    "training-readiness.json"
)
V26_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-"
    "readiness.json"
)


def _v21e_multiseed_config_relative(seed: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
        f"seed{int(seed)}-config.json"
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
        if config.variant == V21A_INTERFRAME_VARIANT:
            experiment_path = _resolve(
                root, config.data.feature_experiment_contract_path or ""
            )
            experiment_expected = (
                config.data.expected_feature_experiment_contract_sha256
            )
            manifest_caches = dict(manifest.get("cache_files") or {})
            manifest_train = dict(manifest_caches.get("train") or {})
            manifest_validation = dict(manifest_caches.get("val") or {})
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or manifest.get("schema_version") != KNOWN_GATE_V21A_MANIFEST_SCHEMA
                or manifest.get("feature_schema")
                != "cq08-current-event-interframe-spectral-consistency-v1"
                or manifest.get("core_data_contract_sha256") != expected["contract"]
                or manifest.get("experiment_contract_sha256") != experiment_expected
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
                or manifest.get("validation_label_values_used_for_readiness_gate")
                is not False
                or _resolve(root, str(manifest_train.get("path", "")))
                != cache_paths["train"]
                or manifest_train.get("sha256") != cache_expected["train"]
                or _resolve(root, str(manifest_validation.get("path", "")))
                != cache_paths["validation"]
                or manifest_validation.get("sha256") != cache_expected["validation"]
            ):
                raise ValueError(
                    "V21a inter-frame feature manifest/experiment contract is stale"
                )
        elif config.variant == V21D_PREFIX_VARIANT:
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
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or manifest.get("schema_version") != KNOWN_GATE_V21D_MANIFEST_SCHEMA
                or manifest.get("feature_schema") != D_RANGE_PREFIX_FEATURE_SCHEMA
                or manifest.get("core_data_contract_sha256") != expected["contract"]
                or manifest.get("experiment_contract_sha256") != experiment_expected
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
                or _resolve(root, str(manifest_train.get("path", "")))
                != cache_paths["train"]
                or manifest_train.get("sha256") != cache_expected["train"]
                or _resolve(root, str(manifest_validation.get("path", "")))
                != cache_paths["validation"]
                or manifest_validation.get("sha256") != cache_expected["validation"]
                or experiment.get("schema_version")
                != (
                    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-"
                    "clutter-bird-experiment-contract-v1"
                )
                or experiment.get("internal_variant") != V21D_PREFIX_VARIANT
                or experiment.get("feature_names") != list(config.feature_names)
                or dict(experiment.get("evaluation") or {}).get(
                    "primary_hypothesis_labels"
                )
                != ["BirdTarget", "ClutterTarget"]
                or dict(experiment.get("evaluation") or {}).get("safety_labels")
                != ["DroneTarget", "BalloonTarget"]
                or experiment.get("core_data_contract_sha256") != expected["contract"]
                or not experiment_document_path.is_file()
                or sha256_file(experiment_document_path)
                != experiment.get("experiment_document_sha256")
                or experiment.get("test") != "未执行"
                or int(experiment.get("test_payload_access_count", -1)) != 0
            ):
                raise ValueError(
                    "V21d causal-prefix d-range manifest/experiment contract is stale"
                )
        elif config.variant in {
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
        elif config.variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
            experiment_path = _resolve(
                root, config.data.feature_experiment_contract_path or ""
            )
            experiment_expected = config.data.expected_feature_experiment_contract_sha256
            experiment = (
                json.loads(experiment_path.read_text(encoding="utf-8"))
                if experiment_path.is_file()
                else {}
            )
            experiment_data = dict(experiment.get("data_contract") or {})
            experiment_cache = dict(experiment.get("feature_cache_reuse") or {})
            experiment_treatment = dict(experiment.get("treatment") or {})
            experiment_training = dict(experiment.get("training") or {})
            manifest_caches = dict(manifest.get("cache_files") or {})
            manifest_train = dict(manifest_caches.get("train") or {})
            manifest_validation = dict(manifest_caches.get("val") or {})
            source_manifest_path = _resolve(
                root, str(manifest.get("source_manifest", ""))
            )
            source_manifest_expected = str(
                manifest.get("source_manifest_sha256", "")
            )
            source_manifest = (
                json.loads(source_manifest_path.read_text(encoding="utf-8"))
                if source_manifest_path.is_file()
                else {}
            )
            source_caches = dict(source_manifest.get("cache_files") or {})
            source_train = dict(source_caches.get("train") or {})
            source_validation = dict(source_caches.get("val") or {})
            source_experiment_path = _resolve(
                root, str(experiment_cache.get("contract", ""))
            )
            source_experiment_expected = str(
                experiment_cache.get("contract_sha256", "")
            )
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or manifest.get("schema_version") != KNOWN_GATE_V22A_MANIFEST_SCHEMA
                or manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
                or manifest.get("feature_names") != list(config.feature_names)
                or _resolve(root, str(manifest.get("experiment_contract", "")))
                != experiment_path
                or manifest.get("experiment_contract_sha256") != experiment_expected
                or not source_manifest_path.is_file()
                or sha256_file(source_manifest_path) != source_manifest_expected
                or source_manifest.get("schema_version")
                != KNOWN_GATE_V21E_MANIFEST_SCHEMA
                or _resolve(root, str(source_manifest.get("core_data_contract", "")))
                != paths["contract"]
                or source_manifest.get("core_data_contract_sha256")
                != expected["contract"]
                or not source_experiment_path.is_file()
                or sha256_file(source_experiment_path) != source_experiment_expected
                or _resolve(
                    root, str(source_manifest.get("experiment_contract", ""))
                )
                != source_experiment_path
                or source_manifest.get("experiment_contract_sha256")
                != source_experiment_expected
                or _resolve(root, str(source_train.get("path", "")))
                != cache_paths["train"]
                or source_train.get("sha256") != cache_expected["train"]
                or _resolve(root, str(source_validation.get("path", "")))
                != cache_paths["validation"]
                or source_validation.get("sha256") != cache_expected["validation"]
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
                or manifest.get("sample_id_is_model_input") is not False
                or manifest.get("internal_rd_residual_is_not_cache_field") is not True
                or manifest.get("unknown_excluded") is not True
                or _resolve(root, str(manifest_train.get("path", ""))) != cache_paths["train"]
                or manifest_train.get("sha256") != cache_expected["train"]
                or _resolve(root, str(manifest_validation.get("path", ""))) != cache_paths["validation"]
                or manifest_validation.get("sha256") != cache_expected["validation"]
                or experiment.get("schema_version") != KNOWN_GATE_V22A_EXPERIMENT_SCHEMA
                or experiment.get("internal_variant")
                != V22A_RD_TEMPORAL_RESIDUAL_VARIANT
                or experiment.get("unknown_training_exposure") != 0
                or experiment.get("validation_used_for_readiness_gate") is not False
                or experiment.get("test_payload_access_count") != 0
                or experiment.get("test") != "未执行"
                or experiment.get("deployment_eligible") is not False
                or _resolve(root, str(experiment_data.get("index_reused", "")))
                != paths["index"]
                or _resolve(root, str(experiment_data.get("core_contract_reused", "")))
                != paths["contract"]
                or _resolve(root, str(experiment_data.get("normalization_reused", "")))
                != paths["stats"]
                or experiment_data.get("rd_tensor_rebuilt") is not False
                or experiment_data.get("new_input_fields") != 0
                or experiment_data.get("test_payload_access_count") != 0
                or _resolve(root, str(experiment_cache.get("manifest", "")))
                != source_manifest_path
                or experiment_cache.get("manifest_sha256")
                != source_manifest_expected
                or _resolve(root, str(experiment_cache.get("train_cache", "")))
                != cache_paths["train"]
                or experiment_cache.get("train_cache_sha256")
                != cache_expected["train"]
                or _resolve(root, str(experiment_cache.get("validation_cache", "")))
                != cache_paths["validation"]
                or experiment_cache.get("validation_cache_sha256")
                != cache_expected["validation"]
                or experiment_treatment.get("new_branch")
                != "last_active_rd_frame_embedding - first_active_rd_frame_embedding"
                or int(experiment_treatment.get("rd_temporal_residual_dim", -1))
                != config.model.rd_temporal_residual_dim
                or experiment_treatment.get("fusion_columns_zero_initialized") is not True
                or experiment_treatment.get("missing_or_single_frame_residual")
                != "exact_zero"
                or int(experiment_training.get("epochs", -1)) != config.train.epochs
                or int(experiment_training.get("seed", -1)) != config.train.seed
                or int(experiment_training.get("dataloader_seed", -1))
                != config.train.dataloader_seed
                or experiment_training.get("loss_mode") != config.train.loss_mode
                or experiment_training.get("class_weight_normalization")
                != config.train.class_weight_normalization
                or experiment_training.get("checkpoint_selection")
                != config.train.checkpoint_selection_metric
                or experiment_training.get("full_parameter_training") is not True
            ):
                raise ValueError("V22a RD temporal residual manifest/experiment contract is stale")
        elif config.variant == V21B_VARIANT:
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
            experiment_document_path = Path(
                str(experiment.get("experiment_document", ""))
            ).resolve()
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or experiment.get("schema_version")
                != (
                    "RADAR-LN3-V21b-known-gate-modal-consistency-"
                    "clutter-drone-experiment-contract-v2"
                )
                or experiment.get("internal_variant") != V21B_VARIANT
                or experiment.get("feature_names") != list(config.feature_names)
                or dict(experiment.get("evaluation") or {}).get(
                    "primary_hypothesis_labels"
                )
                != ["DroneTarget", "ClutterTarget"]
                or dict(experiment.get("evaluation") or {}).get("safety_labels")
                != ["BirdTarget", "BalloonTarget"]
                or experiment.get("core_data_contract_sha256") != expected["contract"]
                or experiment.get("core_feature_manifest_sha256")
                != cache_expected["manifest"]
                or not experiment_document_path.is_file()
                or sha256_file(experiment_document_path)
                != experiment.get("experiment_document_sha256")
                or experiment.get("test") != "未执行"
                or int(experiment.get("test_payload_access_count", -1)) != 0
            ):
                raise ValueError(
                    "V21b modal-consistency experiment contract is missing or stale"
                )
            if (
                manifest.get("schema_version") != KNOWN_GATE_MANIFEST_SCHEMA
                or manifest.get("data_contract_sha256") != expected["contract"]
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
            ):
                raise ValueError("V21b core feature manifest violates cache/open-set semantics")
        elif config.variant == V21C_VARIANT:
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
            experiment_document_path = Path(
                str(experiment.get("experiment_document", ""))
            ).resolve()
            if (
                not experiment_path.is_file()
                or sha256_file(experiment_path) != experiment_expected
                or experiment.get("schema_version")
                != (
                    "RADAR-LN3-V21c-known-gate-target-row-iso-"
                    "clutter-drone-experiment-contract-v1"
                )
                or experiment.get("internal_variant") != V21C_VARIANT
                or experiment.get("feature_names") != list(config.feature_names)
                or dict(experiment.get("evaluation") or {}).get(
                    "primary_hypothesis_labels"
                )
                != ["DroneTarget", "ClutterTarget"]
                or dict(experiment.get("evaluation") or {}).get("safety_labels")
                != ["BirdTarget", "BalloonTarget"]
                or experiment.get("core_data_contract_sha256") != expected["contract"]
                or experiment.get("core_feature_manifest_sha256")
                != cache_expected["manifest"]
                or not experiment_document_path.is_file()
                or sha256_file(experiment_document_path)
                != experiment.get("experiment_document_sha256")
                or experiment.get("test") != "未执行"
                or int(experiment.get("test_payload_access_count", -1)) != 0
            ):
                raise ValueError(
                    "V21c target-row iso experiment contract is missing or stale"
                )
            if (
                manifest.get("schema_version") != KNOWN_GATE_MANIFEST_SCHEMA
                or manifest.get("data_contract_sha256") != expected["contract"]
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_payload_access_count", -1)) != 0
                or manifest.get("valid_mask_is_model_input") is not False
            ):
                raise ValueError(
                    "V21c core feature manifest violates cache/open-set semantics"
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
    if config.variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
        path = (root / V22A_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V22a formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        control_binding = dict(report.get("control") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        reused_hashes = dict(report.get("reused_cache_hashes") or {})
        control_config_path = Path(
            str(control_binding.get("config_path", ""))
        ).resolve()
        control_summary_path = Path(
            str(control_binding.get("summary_path", ""))
        ).resolve()
        control_checkpoint_path = Path(
            str(control_binding.get("checkpoint_path", ""))
        ).resolve()
        control_summary = (
            json.loads(control_summary_path.read_text(encoding="utf-8"))
            if control_summary_path.is_file()
            else {}
        )
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V22A_READINESS_SCHEMA
            or not checks
            or not all(value is True for value in checks.values())
            or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or control_config_path
            != _resolve(root, config.train.paired_baseline_config_path or "")
            or not control_config_path.is_file()
            or control_binding.get("config_sha256")
            != sha256_file(control_config_path)
            or not control_summary_path.is_file()
            or control_binding.get("summary_sha256")
            != sha256_file(control_summary_path)
            or control_summary.get("status") != "completed"
            or control_summary.get("variant") != V21E_COURSE_PREFIX_VARIANT
            or int(control_summary.get("seed", -1)) != config.train.seed
            or not control_checkpoint_path.is_file()
            or control_binding.get("checkpoint_sha256")
            != sha256_file(control_checkpoint_path)
            or Path(str(report.get("contract_path", ""))).resolve()
            != Path(str(cache_evidence.get("experiment_contract_path", ""))).resolve()
            or report.get("contract_sha256")
            != cache_evidence.get("experiment_contract_sha256")
            or Path(str(report.get("feature_manifest_path", ""))).resolve()
            != Path(str(cache_paths.get("manifest", ""))).resolve()
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or reused_hashes.get("train") != cache_hashes.get("train")
            or reused_hashes.get("validation") != cache_hashes.get("validation")
            or int(report.get("rd_temporal_residual_dim", -1))
            != config.model.rd_temporal_residual_dim
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or report.get("test_payload_access_count") != 0
            or report.get("formal_training") != "未执行"
            or report.get("test") != "未执行"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError("V22a readiness is missing, stale, failed or outside the sealed contract")
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V22A_READINESS_SCHEMA,
            "status": "passed",
            "control_summary_path": str(control_summary_path),
            "control_initial_model_state_sha256": str(
                control_summary.get("initial_model_state_sha256", "")
            ),
            "test": "未执行",
        }
    if config.variant == V24_COURSE_REASON_AWARE_BASELINE_VARIANT:
        path = (root / V24_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V24 formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        source = dict(report.get("initialization_source") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        source_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        source_summary_path = _resolve(root, str(source.get("summary_path", "")))
        source_summary = (
            json.loads(source_summary_path.read_text(encoding="utf-8"))
            if source_summary_path.is_file()
            else {}
        )
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V24_READINESS_SCHEMA
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(expected_hashes.get(key) != value for key, value in actual_required.items())
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or source_config_path != _resolve(root, str(source.get("config_path", "")))
            or not source_config_path.is_file()
            or source.get("config_sha256") != sha256_file(source_config_path)
            or not source_summary_path.is_file()
            or source.get("summary_sha256") != sha256_file(source_summary_path)
            or source_summary.get("status") != "completed"
            or source_summary.get("variant") != BASELINE_VARIANT
            or int(source_summary.get("seed", -1)) != 43
            or source.get("initial_model_state_sha256")
            != source_summary.get("initial_model_state_sha256")
            or not str(source.get("initial_model_state_sha256", ""))
            or report.get("initial_model_state_sha256")
            != report.get("v21e_treatment_initial_model_state_sha256")
            or not str(report.get("initial_model_state_sha256", ""))
            or dict(report.get("paired_initialization") or {}).get(
                "fresh_v21_seed43_baseline_initial_model_state_sha256"
            )
            != source.get("initial_model_state_sha256")
            or dict(report.get("paired_initialization") or {}).get(
                "v24_initial_model_state_sha256"
            )
            != report.get("initial_model_state_sha256")
            or source.get("trained_checkpoint_loaded") is not False
            or source.get("trained_checkpoint_accessed") is not False
            or source.get("fresh_random_initialization_reconstructed") is not True
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V24 readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V24_READINESS_SCHEMA,
            "status": "passed",
            "selected_baseline_seed": 43,
            "fresh_training_not_finetune": True,
            "initial_model_state_sha256": str(
                report.get("initial_model_state_sha256", "")
            ),
            "v21e_treatment_initial_model_state_sha256": str(
                report.get("v21e_treatment_initial_model_state_sha256", "")
            ),
            "paired_seed43_baseline_initial_model_state_sha256": str(
                source.get("initial_model_state_sha256", "")
            ),
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V25D_PRESENT_CLASS_MEAN_VARIANT:
        path = (root / V25D_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V25d formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        source = dict(report.get("source_v25a") or {})
        loss_semantics = dict(report.get("loss_semantics_audit") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        control_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        control_summary_path = _resolve(root, str(source.get("summary", "")))
        control_summary = (
            json.loads(control_summary_path.read_text(encoding="utf-8"))
            if control_summary_path.is_file()
            else {}
        )
        source_orders = list(
            dict(control_summary.get("dataloader_contract") or {}).get(
                "epoch_parent_order_hashes"
            )
            or ()
        )
        expected_orders = {
            str(index): str(row.get("sha256", ""))
            for index, row in enumerate(source_orders, start=1)
        }
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V25D_READINESS_SCHEMA
            or report.get("variant") != V25D_PRESENT_CLASS_MEAN_VARIANT
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(
                expected_hashes.get(key) != value
                for key, value in actual_required.items()
            )
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or control_config_path
            != _resolve(
                root,
                "config/RADAR-LN3/RADAR-LN3-V25a-cq08-liquid-known-gate-course-"
                "preclip-gradient-telemetry-seed43-config.json",
            )
            or not control_config_path.is_file()
            or source.get("config_sha256") != sha256_file(control_config_path)
            or not control_summary_path.is_file()
            or source.get("summary_sha256") != sha256_file(control_summary_path)
            or control_summary.get("status") != "completed"
            or control_summary.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
            or int(control_summary.get("seed", -1)) != config.train.seed
            or len(list(control_summary.get("history") or ())) != 20
            or len(source_orders) != 20
            or source.get("initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or report.get("source_v25a_initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or not str(control_summary.get("initial_model_state_sha256", ""))
            or dict(source.get("epoch_parent_order_hashes") or {}) != expected_orders
            or dict(report.get("source_v25a_epoch_parent_order_hashes") or {})
            != expected_orders
            or any(not value for value in expected_orders.values())
            or source.get("trained_checkpoint_loaded") is not False
            or source.get("trained_checkpoint_accessed") is not False
            or loss_semantics.get("mixed_final_effective_class_mass") != [0.5, 0.5]
            or loss_semantics.get("mixed_prefix_effective_class_mass") != [0.5, 0.5]
            or not all(
                bool(value)
                for value in dict(
                    loss_semantics.get("pure_class_loss_equivalent_to_batch_weight_sum")
                    or {}
                ).values()
            )
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V25d readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V25D_READINESS_SCHEMA,
            "status": "passed",
            "source_v25a_summary_path": str(control_summary_path),
            "source_v25a_summary_sha256": str(source.get("summary_sha256", "")),
            "source_v25a_initial_model_state_sha256": str(
                control_summary.get("initial_model_state_sha256", "")
            ),
            "source_v25a_epoch_parent_order_hashes": expected_orders,
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V25E_SHARED_GRADIENT_PROJECTION_VARIANT:
        path = (root / V25E_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V25e formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        source = dict(report.get("source_v25a") or {})
        mechanism = dict(report.get("projection_mechanism") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        control_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        control_summary_path = _resolve(root, str(source.get("summary", "")))
        control_summary = (
            json.loads(control_summary_path.read_text(encoding="utf-8"))
            if control_summary_path.is_file()
            else {}
        )
        source_orders = list(
            dict(control_summary.get("dataloader_contract") or {}).get(
                "epoch_parent_order_hashes"
            )
            or ()
        )
        expected_orders = {
            str(index): str(row.get("sha256", ""))
            for index, row in enumerate(source_orders, start=1)
        }
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V25E_READINESS_SCHEMA
            or report.get("variant") != V25E_SHARED_GRADIENT_PROJECTION_VARIANT
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(expected_hashes.get(key) != value for key, value in actual_required.items())
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or control_config_path
            != _resolve(
                root,
                "config/RADAR-LN3/RADAR-LN3-V25a-cq08-liquid-known-gate-course-"
                "preclip-gradient-telemetry-seed43-config.json",
            )
            or not control_config_path.is_file()
            or source.get("config_sha256") != sha256_file(control_config_path)
            or not control_summary_path.is_file()
            or source.get("summary_sha256") != sha256_file(control_summary_path)
            or control_summary.get("status") != "completed"
            or control_summary.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
            or int(control_summary.get("seed", -1)) != config.train.seed
            or int(control_summary.get("global_step", -1)) != 9860
            or len(list(control_summary.get("history") or ())) != 20
            or len(source_orders) != 20
            or source.get("initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or report.get("source_v25a_initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or not str(control_summary.get("initial_model_state_sha256", ""))
            or dict(source.get("epoch_parent_order_hashes") or {}) != expected_orders
            or dict(report.get("source_v25a_epoch_parent_order_hashes") or {})
            != expected_orders
            or any(not value for value in expected_orders.values())
            or source.get("trained_checkpoint_loaded") is not False
            or source.get("trained_checkpoint_accessed") is not False
            or mechanism.get("projected_modules") != list(V25E_PROJECTED_MODULE_NAMES)
            or mechanism.get("unchanged_modules") != list(V25E_UNCHANGED_MODULE_NAMES)
            or int(mechanism.get("expected_total_updates", -1)) != 9860
            or int(mechanism.get("expected_mixed_batch_count", -1)) != 638
            or int(mechanism.get("expected_pure_batch_count", -1)) != 9222
            or mechanism.get("pure_batches_are_historical_equivalent") is not True
            or mechanism.get("rd_and_classifier_are_unchanged") is not True
            or mechanism.get("projection_is_only_on_negative_conflict") is not True
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V25e readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V25E_READINESS_SCHEMA,
            "status": "passed",
            "source_v25a_summary_path": str(control_summary_path),
            "source_v25a_summary_sha256": str(source.get("summary_sha256", "")),
            "source_v25a_initial_model_state_sha256": str(
                control_summary.get("initial_model_state_sha256", "")
            ),
            "source_v25a_epoch_parent_order_hashes": expected_orders,
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V25A_GRADIENT_TELEMETRY_VARIANT:
        path = (root / V25A_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V25a formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        source = dict(report.get("source_v24") or {})
        source_reference = dict(source.get("fresh_initialization_reference") or {})
        telemetry_schema = dict(report.get("telemetry_schema") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        source_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        sealed_v24_config_path = _resolve(root, str(source.get("config", "")))
        sealed_v24_readiness_path = _resolve(root, str(source.get("readiness", "")))
        sealed_v24_manifest_path = _resolve(
            root, str(source.get("feature_manifest", ""))
        )
        source_reference_summary_path = _resolve(
            root, str(source_reference.get("summary", ""))
        )
        source_reference_summary = (
            json.loads(source_reference_summary_path.read_text(encoding="utf-8"))
            if source_reference_summary_path.is_file()
            else {}
        )
        source_summary_path = _resolve(root, str(source.get("summary", "")))
        source_summary = (
            json.loads(source_summary_path.read_text(encoding="utf-8"))
            if source_summary_path.is_file()
            else {}
        )
        source_orders = list(
            dict(source_summary.get("dataloader_contract") or {}).get(
                "epoch_parent_order_hashes"
            )
            or ()
        )
        expected_orders = {
            str(index): str(row.get("sha256", ""))
            for index, row in enumerate(source_orders, start=1)
        }
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V25A_READINESS_SCHEMA
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(
                expected_hashes.get(key) != value
                for key, value in actual_required.items()
            )
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or source_config_path != _resolve(
                root, "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
            )
            or not source_config_path.is_file()
            or sealed_v24_config_path
            != _resolve(
                root,
                "config/RADAR-LN3/RADAR-LN3-V24-cq08-liquid-known-gate-course-"
                "reason-aware-baseline-seed43-config.json",
            )
            or not sealed_v24_config_path.is_file()
            or source.get("config_sha256") != sha256_file(sealed_v24_config_path)
            or not sealed_v24_readiness_path.is_file()
            or source.get("readiness_sha256") != sha256_file(sealed_v24_readiness_path)
            or not sealed_v24_manifest_path.is_file()
            or source.get("feature_manifest_sha256")
            != sha256_file(sealed_v24_manifest_path)
            or _resolve(root, str(source_reference.get("config", "")))
            != source_config_path
            or source_reference.get("config_sha256") != sha256_file(source_config_path)
            or not source_reference_summary_path.is_file()
            or source_reference.get("summary_sha256")
            != sha256_file(source_reference_summary_path)
            or source_reference_summary.get("status") != "completed"
            or source_reference_summary.get("variant") != BASELINE_VARIANT
            or int(source_reference_summary.get("seed", -1)) != 43
            or len(list(source_reference_summary.get("history") or ())) != 20
            or source_reference.get("initial_model_state_sha256")
            != source_reference_summary.get("initial_model_state_sha256")
            or not str(source_reference.get("initial_model_state_sha256", ""))
            or not source_summary_path.is_file()
            or source.get("summary_sha256") != sha256_file(source_summary_path)
            or source_summary.get("status") != "completed"
            or source_summary.get("variant")
            != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
            or int(source_summary.get("seed", -1)) != 43
            or len(list(source_summary.get("history") or ())) != 20
            or len(source_orders) != 20
            or source.get("initial_model_state_sha256")
            != source_summary.get("initial_model_state_sha256")
            or not str(source.get("initial_model_state_sha256", ""))
            or dict(source.get("epoch_parent_order_hashes") or {}) != expected_orders
            or any(not value for value in expected_orders.values())
            or source.get("trained_checkpoint_loaded") is not False
            or source.get("trained_checkpoint_accessed") is not False
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or telemetry_schema.get("schema_version")
            != "RADAR-LN3-V25a-known-gate-preclip-gradient-batch-telemetry-v1"
            or telemetry_schema.get("capture_timing")
            != "after_loss_backward_before_existing_global_norm_clip"
            or telemetry_schema.get("module_partition")
            != ["tr_encoder", "rd_encoder", "fusion", "liquid", "classifier"]
            or telemetry_schema.get("raw_parent_sample_ids_stored") is not False
            or telemetry_schema.get("changes_gradients") is not False
            or telemetry_schema.get("changes_optimizer") is not False
            or telemetry_schema.get("changes_loss") is not False
            or telemetry_schema.get("changes_data_order") is not False
            or telemetry_schema.get("epoch_boundary_resume_only") is not True
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V25a readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V25A_READINESS_SCHEMA,
            "status": "passed",
            "source_v24_summary_path": str(source_summary_path),
            "source_v24_summary_sha256": str(source.get("summary_sha256", "")),
            "source_v24_initial_model_state_sha256": str(
                source.get("initial_model_state_sha256", "")
            ),
            "paired_v21_seed43_baseline_initial_model_state_sha256": str(
                source_reference.get("initial_model_state_sha256", "")
            ),
            "source_v24_epoch_parent_order_hashes": expected_orders,
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT:
        path = (root / V24A_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V24a formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        control = dict(report.get("control") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        control_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        control_summary_path = _resolve(root, str(control.get("summary", "")))
        control_summary = (
            json.loads(control_summary_path.read_text(encoding="utf-8"))
            if control_summary_path.is_file()
            else {}
        )
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V24A_READINESS_SCHEMA
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(
                expected_hashes.get(key) != value
                for key, value in actual_required.items()
            )
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or control_config_path
            != _resolve(root, str(control.get("config", "")))
            or not control_config_path.is_file()
            or control.get("config_sha256") != sha256_file(control_config_path)
            or not control_summary_path.is_file()
            or control.get("summary_sha256") != sha256_file(control_summary_path)
            or control_summary.get("status") != "completed"
            or control_summary.get("variant")
            != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
            or int(control_summary.get("seed", -1)) != config.train.seed
            or control.get("initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or not str(control.get("initial_model_state_sha256", ""))
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V24a readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V24A_READINESS_SCHEMA,
            "status": "passed",
            "control_initial_model_state_sha256": str(
                control.get("initial_model_state_sha256", "")
            ),
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT:
        path = (root / V24C_READINESS_RELATIVE).resolve()
        if not path.is_file():
            raise PermissionError(f"V24c formal training requires readiness: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        binding = dict(report.get("config") or {})
        checks = dict(report.get("checks") or {})
        control = dict(report.get("control") or {})
        cache_evidence = dict(contract_evidence.get("feature_cache") or {})
        cache_paths = dict(cache_evidence.get("paths") or {})
        cache_hashes = dict(cache_evidence.get("hashes") or {})
        experiment_contract_path = _resolve(
            root, config.data.feature_experiment_contract_path or ""
        )
        control_config_path = _resolve(
            root, config.train.paired_baseline_config_path or ""
        )
        control_summary_path = _resolve(root, str(control.get("summary", "")))
        control_summary = (
            json.loads(control_summary_path.read_text(encoding="utf-8"))
            if control_summary_path.is_file()
            else {}
        )
        actual_required = {
            "index_sha256": contract_evidence["hashes"]["index_sha256"],
            "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
            "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
            "feature_manifest_sha256": cache_hashes.get("manifest"),
            "train_feature_cache_sha256": cache_hashes.get("train"),
            "validation_feature_cache_sha256": cache_hashes.get("validation"),
            "experiment_contract_sha256": contract_evidence["hashes"].get(
                "feature_experiment_contract_sha256"
            ),
        }
        expected_hashes = dict(report.get("contract_hashes") or {})
        initialization = dict(report.get("initialization") or {})
        if (
            report.get("status") != "passed"
            or report.get("schema_version") != KNOWN_GATE_V24C_READINESS_SCHEMA
            or not checks
            or not all(value is True for value in checks.values())
            or _resolve(root, str(binding.get("path", ""))) != config_path.resolve()
            or binding.get("sha256") != sha256_file(config_path)
            or any(
                expected_hashes.get(key) != value
                for key, value in actual_required.items()
            )
            or _resolve(root, str(report.get("feature_manifest_path", "")))
            != _resolve(root, str(cache_paths.get("manifest", "")))
            or report.get("feature_manifest_sha256") != cache_hashes.get("manifest")
            or _resolve(root, str(report.get("contract_path", "")))
            != experiment_contract_path
            or report.get("contract_sha256")
            != actual_required["experiment_contract_sha256"]
            or control_config_path
            != _resolve(root, str(control.get("config", "")))
            or not control_config_path.is_file()
            or control.get("config_sha256") != sha256_file(control_config_path)
            or not control_summary_path.is_file()
            or control.get("summary_sha256") != sha256_file(control_summary_path)
            or control_summary.get("status") != "completed"
            or control_summary.get("variant")
            != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
            or int(control_summary.get("seed", -1)) != config.train.seed
            or control.get("initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or not str(control.get("initial_model_state_sha256", ""))
            or initialization.get("v24_control_initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or initialization.get("v24c_initial_model_state_sha256")
            != control_summary.get("initial_model_state_sha256")
            or initialization.get("only_dynamics_mode_differs") is not True
            or initialization.get("trained_v24_checkpoint_loaded_for_initialization")
            is not False
            or int(report.get("known_train_parent_count", -1)) != 986
            or int(report.get("known_validation_parent_count", -1)) != 232
            or int(report.get("validation_clutter_parent_support", -1)) != 9
            or int(report.get("unknown_training_exposure", -1)) != 0
            or report.get("validation_used_for_readiness_gate") is not False
            or int(report.get("test_payload_access_count", -1)) != 0
            or report.get("formal_training") != "not_executed"
            or report.get("OOF") != "not_executed"
            or report.get("test") != "not_executed"
            or report.get("formal_rejection") != "not_executed"
            or report.get("full_cascade") != "not_executed"
            or report.get("stable") != "not_executed"
            or report.get("P95") != "not_executed"
            or report.get("deployment_eligible") is not False
        ):
            raise PermissionError(
                "V24c readiness is missing, stale, failed or outside the sealed contract"
            )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": KNOWN_GATE_V24C_READINESS_SCHEMA,
            "status": "passed",
            "control_initial_model_state_sha256": str(
                control.get("initial_model_state_sha256", "")
            ),
            "validation_used_for_readiness_gate": False,
            "test": "not_executed",
        }
    if config.variant == V21A_VARIANT:
        raise PermissionError(
            "the seed42 V21a intra-frame STFT candidate is superseded; use the "
            "v21a_interframe seed43 selected-Baseline experiment"
        )
    if config.variant == V21B_VARIANT and config.train.seed != 43:
        raise PermissionError(
            "the seed42 V21b candidate is not comparable to the selected Baseline; "
            "use the seed43 modal-consistency experiment"
        )
    if config.variant == V21C_VARIANT and config.train.seed != 43:
        raise PermissionError(
            "the seed42 V21c candidate is not comparable to the selected Baseline; "
            "use the seed43 target-row iso experiment"
        )
    if config.variant == V21D_VARIANT:
        raise PermissionError(
            "the original V21d current-event proxy does not implement the historical "
            "d-range statistic; use v21d_prefix seed43"
        )
    if config.variant == V21D_PREFIX_VARIANT and config.train.seed != 43:
        raise PermissionError("V21d must pair with selected Baseline seed43")
    if (
        config.variant == V21B_VARIANT
        and config_path.resolve() != (root / V21B_FORMAL_CONFIG_RELATIVE).resolve()
    ):
        raise PermissionError(
            "the earlier V21b seed43 contract used a broader/incorrect direction "
            "claim; use the frozen Clutter/Drone-only modal-consistency config"
        )
    if (
        config.variant == V21C_VARIANT
        and config_path.resolve() != (root / V21C_FORMAL_CONFIG_RELATIVE).resolve()
    ):
        raise PermissionError(
            "the earlier V21c seed42 draft is non-comparable; use the frozen "
            "selected-seed43 Clutter/Drone target-row iso config"
        )
    if (
        config.variant == V21D_PREFIX_VARIANT
        and config_path.resolve() != (root / V21D_FORMAL_CONFIG_RELATIVE).resolve()
    ):
        raise PermissionError(
            "use the frozen selected-seed43 Clutter/Bird causal-prefix d-range config"
        )
    if config.variant == V21E_COURSE_PREFIX_VARIANT:
        seed = int(config.train.seed)
        selected_seed43 = (
            seed == 43
            and config_path.resolve() == (root / V21E_FORMAL_CONFIG_RELATIVE).resolve()
        )
        multiseed_confirmation = (
            seed in {42, 44}
            and config_path.resolve()
            == (root / _v21e_multiseed_config_relative(seed)).resolve()
        )
        if not (selected_seed43 or multiseed_confirmation):
            raise PermissionError(
                "use the frozen seed43 V21e config or the sealed seed42/44 "
                "multiseed confirmation config"
            )
    if config.variant == V21A_INTERFRAME_VARIANT:
        return _verify_v21a_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
    if config.variant == V21B_VARIANT:
        return _verify_v21b_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
    if config.variant == V21C_VARIANT:
        return _verify_v21c_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
    if config.variant == V21D_PREFIX_VARIANT:
        return _verify_v21d_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
    if config.variant == V21E_COURSE_PREFIX_VARIANT:
        if config.train.seed in {42, 44}:
            return _verify_v21e_multiseed_readiness(
                root,
                config_path,
                config,
                contract_evidence,
            )
        return _verify_v21e_readiness(
            root,
            config_path,
            config,
            contract_evidence,
        )
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


def _verify_v21a_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if config.train.seed != 43:
        raise PermissionError("corrected V21a must pair with selected Baseline seed43")
    path = (root / V21A_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21a formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(report.get("config") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    selected = dict(report.get("selected_baseline") or {})
    baseline_config_path = _resolve(
        root, config.train.paired_baseline_config_path or ""
    )
    selection_path = Path(str(selected.get("selection_path", ""))).resolve()
    baseline_summary_path = Path(str(selected.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(selected.get("checkpoint_path", ""))).resolve()
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_V21A_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or int(selected.get("seed", -1)) != 43
        or Path(str(selected.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or selected.get("config_sha256") != sha256_file(baseline_config_path)
        or not selection_path.is_file()
        or selected.get("selection_sha256") != sha256_file(selection_path)
        or not baseline_summary_path.is_file()
        or selected.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or selected.get("checkpoint_sha256")
        != sha256_file(baseline_checkpoint_path)
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21a readiness is missing, stale, failed or not bound to selected seed43"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21A_READINESS_SCHEMA,
        "status": "passed",
        "selected_baseline_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def _verify_v21b_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    path = (root / V21B_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21b formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(report.get("config") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    selected = dict(report.get("selected_baseline") or {})
    baseline_config_path = _resolve(
        root, config.train.paired_baseline_config_path or ""
    )
    selection_path = Path(str(selected.get("selection_path", ""))).resolve()
    baseline_summary_path = Path(str(selected.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(selected.get("checkpoint_path", ""))).resolve()
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_V21B_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or int(selected.get("seed", -1)) != 43
        or Path(str(selected.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or selected.get("config_sha256") != sha256_file(baseline_config_path)
        or not selection_path.is_file()
        or selected.get("selection_sha256") != sha256_file(selection_path)
        or not baseline_summary_path.is_file()
        or selected.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or selected.get("checkpoint_sha256")
        != sha256_file(baseline_checkpoint_path)
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21b readiness is missing, stale, failed or not bound to selected seed43"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21B_READINESS_SCHEMA,
        "status": "passed",
        "selected_baseline_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def _verify_v21c_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    path = (root / V21C_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21c formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(report.get("config") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    selected = dict(report.get("selected_baseline") or {})
    baseline_config_path = _resolve(
        root, config.train.paired_baseline_config_path or ""
    )
    selection_path = Path(str(selected.get("selection_path", ""))).resolve()
    baseline_summary_path = Path(str(selected.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(selected.get("checkpoint_path", ""))).resolve()
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_V21C_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or int(selected.get("seed", -1)) != 43
        or Path(str(selected.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or selected.get("config_sha256") != sha256_file(baseline_config_path)
        or not selection_path.is_file()
        or selected.get("selection_sha256") != sha256_file(selection_path)
        or not baseline_summary_path.is_file()
        or selected.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or selected.get("checkpoint_sha256")
        != sha256_file(baseline_checkpoint_path)
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21c readiness is missing, stale, failed or not bound to selected seed43"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21C_READINESS_SCHEMA,
        "status": "passed",
        "selected_baseline_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def _verify_v21d_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    path = (root / V21D_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21d formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(report.get("config") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    selected = dict(report.get("selected_baseline") or {})
    baseline_config_path = _resolve(
        root, config.train.paired_baseline_config_path or ""
    )
    selection_path = Path(str(selected.get("selection_path", ""))).resolve()
    baseline_summary_path = Path(str(selected.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(selected.get("checkpoint_path", ""))).resolve()
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_V21D_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or int(selected.get("seed", -1)) != 43
        or Path(str(selected.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or selected.get("config_sha256") != sha256_file(baseline_config_path)
        or not selection_path.is_file()
        or selected.get("selection_sha256") != sha256_file(selection_path)
        or not baseline_summary_path.is_file()
        or selected.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or selected.get("checkpoint_sha256")
        != sha256_file(baseline_checkpoint_path)
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21d readiness is missing, stale, failed or not bound to selected seed43"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21D_READINESS_SCHEMA,
        "status": "passed",
        "selected_baseline_seed": 43,
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def _verify_v21e_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    path = (root / V21E_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21e formal training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(report.get("config") or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    selected = dict(report.get("selected_baseline") or {})
    baseline_config_path = _resolve(
        root, config.train.paired_baseline_config_path or ""
    )
    selection_path = Path(str(selected.get("selection_path", ""))).resolve()
    baseline_summary_path = Path(str(selected.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(selected.get("checkpoint_path", ""))).resolve()
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != KNOWN_GATE_V21E_READINESS_SCHEMA
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or int(selected.get("seed", -1)) != 43
        or Path(str(selected.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or selected.get("config_sha256") != sha256_file(baseline_config_path)
        or not selection_path.is_file()
        or selected.get("selection_sha256") != sha256_file(selection_path)
        or not baseline_summary_path.is_file()
        or selected.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or selected.get("checkpoint_sha256")
        != sha256_file(baseline_checkpoint_path)
        or report.get("post_hoc_feature_discovery") is not True
        or report.get("validation_used_for_feature_selection_before_v21e") is not True
        or report.get("validation_used_for_readiness_gate") is not False
        or report.get("validation_feature_values_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21e readiness is missing, stale, failed or not bound to selected seed43"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21E_READINESS_SCHEMA,
        "status": "passed",
        "selected_baseline_seed": 43,
        "post_hoc_feature_discovery": True,
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
    }


def _verify_v21e_multiseed_readiness(
    root: Path,
    config_path: Path,
    config: KnownGateConfig,
    contract_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(config.train.seed)
    if seed not in {42, 44}:
        raise PermissionError("V21e multiseed training is restricted to seeds 42 and 44")
    path = (root / V21E_MULTISEED_READINESS_RELATIVE).resolve()
    if not path.is_file():
        raise PermissionError(f"V21e multiseed training requires readiness: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(report.get("checks") or {})
    binding = dict(dict(report.get("treatment_configs") or {}).get(str(seed)) or {})
    baseline = dict(dict(report.get("baseline_runs") or {}).get(str(seed)) or {})
    expected_hashes = dict(report.get("contract_hashes") or {})
    actual_required = {
        "index_sha256": contract_evidence["hashes"]["index_sha256"],
        "core_contract_sha256": contract_evidence["hashes"]["contract_sha256"],
        "normalization_sha256": contract_evidence["hashes"]["normalization_sha256"],
        "experiment_contract_sha256": contract_evidence["hashes"][
            "feature_experiment_contract_sha256"
        ],
        "feature_manifest_sha256": contract_evidence["hashes"][
            "feature_manifest_sha256"
        ],
        "train_feature_cache_sha256": contract_evidence["hashes"][
            "train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": contract_evidence["hashes"][
            "validation_feature_cache_sha256"
        ],
    }
    baseline_config_path = _resolve(root, config.train.paired_baseline_config_path or "")
    baseline_summary_path = Path(str(baseline.get("summary_path", ""))).resolve()
    baseline_checkpoint_path = Path(str(baseline.get("checkpoint_path", ""))).resolve()
    if (
        report.get("status") != "passed"
        or report.get("schema_version")
        != KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA
        or tuple(report.get("seed_set") or ()) != KNOWN_GATE_V21E_MULTISEED_SEEDS
        or tuple(report.get("new_training_seeds") or ()) != (42, 44)
        or not checks
        or not all(value is True for value in checks.values())
        or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        or binding.get("sha256") != sha256_file(config_path)
        or int(binding.get("seed", -1)) != seed
        or int(baseline.get("seed", -1)) != seed
        or Path(str(baseline.get("config_path", ""))).resolve()
        != baseline_config_path
        or not baseline_config_path.is_file()
        or baseline.get("config_sha256") != sha256_file(baseline_config_path)
        or not baseline_summary_path.is_file()
        or baseline.get("summary_sha256") != sha256_file(baseline_summary_path)
        or not baseline_checkpoint_path.is_file()
        or baseline.get("checkpoint_sha256") != sha256_file(baseline_checkpoint_path)
        or any(expected_hashes.get(key) != value for key, value in actual_required.items())
        or report.get("post_hoc_feature_discovery") is not True
        or report.get("validation_used_for_readiness_gate") is not False
        or int(report.get("test_payload_access_count", -1)) != 0
        or report.get("formal_training_at_report_generation") != "未执行"
        or report.get("test") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise PermissionError(
            "V21e multiseed readiness is missing, stale, failed or not paired "
            f"to Baseline seed{seed}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
        "status": "passed",
        "paired_baseline_seed": seed,
        "post_hoc_feature_discovery": True,
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





