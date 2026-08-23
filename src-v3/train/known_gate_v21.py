from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
    KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_CONTRACT_SCHEMA,
    KNOWN_GATE_MANIFEST_SCHEMA,
    KNOWN_GATE_READINESS_SCHEMA,
    KNOWN_GATE_SELECTION_METRIC,
    KNOWN_GATE_V21A_MANIFEST_SCHEMA,
    KNOWN_GATE_V21A_READINESS_SCHEMA,
    KNOWN_GATE_V21B_READINESS_SCHEMA,
    KNOWN_GATE_V21C_READINESS_SCHEMA,
    KNOWN_GATE_V21D_MANIFEST_SCHEMA,
    KNOWN_GATE_V21D_READINESS_SCHEMA,
    KNOWN_GATE_V21E_MANIFEST_SCHEMA,
    KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
    KNOWN_GATE_V21E_MULTISEED_SEEDS,
    KNOWN_GATE_V21E_READINESS_SCHEMA,
    KNOWN_GATE_V22A_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V22A_MANIFEST_SCHEMA,
    KNOWN_GATE_V22A_READINESS_SCHEMA,
    KNOWN_GATE_V24_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24_MANIFEST_SCHEMA,
    KNOWN_GATE_V24_READINESS_SCHEMA,
    KNOWN_GATE_V24A_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24A_MANIFEST_SCHEMA,
    KNOWN_GATE_V24A_READINESS_SCHEMA,
    KNOWN_GATE_V24C_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24C_MANIFEST_SCHEMA,
    KNOWN_GATE_V24C_READINESS_SCHEMA,
    V22A_SCHEMA_VERSION,
    V24_SCHEMA_VERSION,
    V24A_SCHEMA_VERSION,
    V24C_SCHEMA_VERSION,
    UNKNOWN_LABEL,
    KnownGateClassifier,
    KnownGateConfig,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    evaluate_known_gate,
    known_gate_config_identity_sha256,
    known_gate_device,
    load_known_gate_config,
    seed_everything,
    sha256_file,
)
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
)
from radar_ln3.logging_utils import (
    close_training_logger,
    configure_training_logger,
    format_duration,
    format_metric,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary


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
            expected_manifest_schema = (
                KNOWN_GATE_V24A_MANIFEST_SCHEMA
                if is_v24a
                else (
                    KNOWN_GATE_V24C_MANIFEST_SCHEMA
                    if is_v24c
                    else (
                        KNOWN_GATE_V24_MANIFEST_SCHEMA
                        if is_v24
                        else KNOWN_GATE_V21E_MANIFEST_SCHEMA
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
                        KNOWN_GATE_V24_EXPERIMENT_SCHEMA
                        if is_v24
                        else (
                            "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
                            "experiment-contract-v1"
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
                        V24_COURSE_REASON_AWARE_BASELINE_VARIANT
                        if is_v24
                        else V21E_COURSE_PREFIX_VARIANT
                    )
                )
            )
            expected_not_executed = (
                "not_executed" if (is_v24 or is_v24a or is_v24c) else "未执行"
            )
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
                    and experiment.get("post_hoc_feature_discovery") is not True
                )
                or (
                    (is_v24 or is_v24a or is_v24c)
                    and experiment.get("fresh_training_not_finetune") is not True
                )
                or experiment.get("test") != expected_not_executed
                or int(experiment.get("test_payload_access_count", -1)) != 0
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
            if config.variant == V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT:
                reproduced_control_state = _state_dict_sha256(
                    baseline_reference.state_dict()
                )
                if (
                    reproduced_control_state
                    != readiness.get("control_initial_model_state_sha256")
                    or int(paired_initialization.get("new_fusion_columns", -1)) != 1
                    or paired_initialization.get("new_fusion_columns_exactly_zero")
                    is not True
                ):
                    raise AssertionError(
                        "V24a current control initialization differs from the sealed V24 protocol"
                    )
                paired_initialization[
                    "sealed_v24_control_initial_model_state_reproduced"
                ] = True
                paired_initialization[
                    "sealed_v24_control_initial_model_state_sha256"
                ] = reproduced_control_state
            if config.variant == V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT:
                reproduced_control_state = _state_dict_sha256(
                    baseline_reference.state_dict()
                )
                if (
                    initial_state_sha256 != reproduced_control_state
                    or reproduced_control_state
                    != readiness.get("control_initial_model_state_sha256")
                    or paired_initialization.get("dynamics_only_pair") is not True
                    or int(paired_initialization.get("new_fusion_columns", -1)) != 0
                    or paired_initialization.get("new_fusion_columns_exactly_zero")
                    is not True
                    or model.liquid.invalid_dt_update_mode != "strict_skip"
                    or baseline_reference.liquid.invalid_dt_update_mode
                    != "bounded_observation"
                ):
                    raise AssertionError(
                        "V24c strict-skip initialization differs beyond the sealed dynamics variable"
                    )
                paired_initialization[
                    "sealed_v24_control_initial_model_state_reproduced"
                ] = True
                paired_initialization[
                    "sealed_v24_control_initial_model_state_sha256"
                ] = reproduced_control_state
            if config.variant == V24_COURSE_REASON_AWARE_BASELINE_VARIANT:
                baseline_initial_state = _state_dict_sha256(
                    baseline_reference.state_dict()
                )
                if (
                    initial_state_sha256
                    != readiness.get("initial_model_state_sha256")
                    or initial_state_sha256
                    != readiness.get(
                        "v21e_treatment_initial_model_state_sha256"
                    )
                    or baseline_initial_state
                    != readiness.get(
                        "paired_seed43_baseline_initial_model_state_sha256"
                    )
                ):
                    raise AssertionError(
                        "V24 current model initialization differs from the sealed fresh baseline protocol"
                    )
                paired_initialization[
                    "sealed_v24_initial_model_state_reproduced"
                ] = True
                paired_initialization[
                    "sealed_v24_initial_model_state_sha256"
                ] = initial_state_sha256
            if config.variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
                reproduced_control_state = _state_dict_sha256(
                    baseline_reference.state_dict()
                )
                if reproduced_control_state != readiness.get(
                    "control_initial_model_state_sha256"
                ):
                    raise AssertionError(
                        "V22a current Control initialization differs from the frozen V21e run"
                    )
                paired_initialization[
                    "frozen_v21e_initial_model_state_reproduced"
                ] = True
                paired_initialization[
                    "frozen_v21e_initial_model_state_sha256"
                ] = reproduced_control_state
            audit_loader = build_known_gate_dataloader(
                train_dataset,
                batch_size=min(config.train.batch_size, len(train_dataset)),
                shuffle=False,
            )
            probe = next(iter(audit_loader))
            same_second_probe: Mapping[str, Any] | None = None
            if config.variant == V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT:
                same_second_probe = probe
                if not bool((same_second_probe["tr_dt_reason"] == 2).any()):
                    for candidate in audit_loader:
                        if bool((candidate["tr_dt_reason"] == 2).any()):
                            same_second_probe = candidate
                            break
                if not bool((same_second_probe["tr_dt_reason"] == 2).any()):
                    raise AssertionError(
                        "V24c train population unexpectedly lacks a same-second event"
                    )
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
            if config.variant == V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT:
                if same_second_probe is None:
                    raise RuntimeError("V24c same-second audit probe was not selected")
                audit_probe = same_second_probe
                with torch.inference_mode():
                    strict_audit_output = model(**_model_inputs(audit_probe))
                    bounded_audit_output = baseline_reference(
                        **_model_inputs(audit_probe)
                    )
                zero_delta = audit_probe["tr_dt_reason"] == 2
                if not bool(zero_delta.any()):
                    raise AssertionError(
                        "V24c real training probe lacks a same-second event for dynamics audit"
                    )
                no_zero_probe = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in audit_probe.items()
                }
                no_zero_probe["tr_dt_seconds"] = no_zero_probe["tr_dt_seconds"].clone()
                no_zero_probe["tr_dt_valid"] = no_zero_probe["tr_dt_valid"].clone()
                no_zero_probe["tr_dt_reason"] = no_zero_probe["tr_dt_reason"].clone()
                no_zero_probe["tr_dt_seconds"][zero_delta] = 1.0
                no_zero_probe["tr_dt_valid"][zero_delta] = True
                no_zero_probe["tr_dt_reason"][zero_delta] = 3
                with torch.inference_mode():
                    strict_no_zero_logits = model(**_model_inputs(no_zero_probe))["sequence_logits"]
                    control_no_zero_logits = baseline_reference(**_model_inputs(no_zero_probe))["sequence_logits"]
                if not torch.equal(strict_no_zero_logits, control_no_zero_logits):
                    raise AssertionError(
                        "V24c/control initial logits differ when no same-second dynamics are present"
                    )
                strict_state = strict_audit_output["liquid_sequence"]
                previous_state = torch.cat(
                    [
                        torch.zeros_like(strict_state[:, :1]),
                        strict_state[:, :-1],
                    ],
                    dim=1,
                )
                if not torch.equal(
                    strict_state.masked_select(zero_delta.unsqueeze(-1)),
                    previous_state.masked_select(zero_delta.unsqueeze(-1)),
                ):
                    raise AssertionError(
                        "V24c strict-skip state changed at a same-second event"
                    )
                paired_initialization.update(
                    {
                        "initial_logits_exactly_equal_without_zero_delta": True,
                        "initial_logits_without_zero_delta_sha256": hashlib.sha256(
                            strict_no_zero_logits.detach()
                            .cpu()
                            .contiguous()
                            .numpy()
                            .tobytes(order="C")
                        ).hexdigest(),
                        "real_training_probe_same_second_event_count": int(
                            zero_delta.sum().item()
                        ),
                        "strict_skip_state_unchanged_at_every_probe_same_second_event": True,
                        "strict_skip_and_bounded_logits_are_allowed_to_differ_with_same_second_events": (
                            not torch.equal(
                                strict_audit_output["sequence_logits"],
                                bounded_audit_output["sequence_logits"],
                            )
                        ),
                    }
                )
                del (
                    strict_no_zero_logits,
                    control_no_zero_logits,
                    strict_audit_output,
                    bounded_audit_output,
                    no_zero_probe,
                )
            else:
                if not torch.equal(treatment_logits, baseline_logits):
                    raise AssertionError("V21 Treatment initial logits differ from the paired Baseline")
                paired_initialization["initial_logits_exactly_equal"] = True
                paired_initialization["initial_logits_sha256"] = hashlib.sha256(
                    treatment_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
                ).hexdigest()
            if config.variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
                residual = treatment_output["rd_temporal_residual"]
                active = ~probe["tr_padding_mask"]
                frame_counts = probe["rd_frame_mask"].sum(dim=2)
                multiple = active & (frame_counts >= 2)
                if not bool(multiple.any()):
                    raise AssertionError(
                        "V22a real training probe contains no multi-frame RD event"
                    )
                residual_l2 = torch.linalg.vector_norm(residual, dim=-1)
                nonzero_multiple = multiple & (residual_l2 > 0.0)
                if not bool(nonzero_multiple.any()):
                    raise AssertionError(
                        "V22a real training probe produces no non-zero RD residual"
                    )
                model.zero_grad(set_to_none=True)
                gradient_output = model(**_model_inputs(probe))
                gradient_objective = torch.nn.functional.cross_entropy(
                    gradient_output["window_logits"], probe["labels"]
                )
                gradient_objective.backward()
                residual_start = model.base_fusion_input_dim
                residual_end = residual_start + model.rd_temporal_residual_dim
                residual_gradient = model.fusion[0].weight.grad[
                    :, residual_start:residual_end
                ]
                if (
                    residual_gradient is None
                    or not torch.isfinite(residual_gradient).all()
                    or float(residual_gradient.abs().max().item()) <= 0.0
                ):
                    raise AssertionError(
                        "V22a real training probe gives no gradient to residual columns"
                    )
                paired_initialization.update(
                    {
                        "real_training_probe_parent_count": len(
                            probe["parent_sample_ids"]
                        ),
                        "real_training_probe_active_event_count": int(
                            active.sum().item()
                        ),
                        "real_training_probe_multiframe_event_count": int(
                            multiple.sum().item()
                        ),
                        "real_training_probe_nonzero_residual_event_count": int(
                            nonzero_multiple.sum().item()
                        ),
                        "real_training_probe_residual_l2_max": float(
                            residual_l2.masked_select(multiple).max().item()
                        ),
                        "real_training_probe_residual_gradient_abs_max": float(
                            residual_gradient.abs().max().item()
                        ),
                        "real_training_probe_residual_columns_receive_gradient": True,
                    }
                )
                model.zero_grad(set_to_none=True)
                del gradient_output, gradient_objective, residual_gradient
            if config.variant == V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT:
                gradient_probe = probe
                for candidate in probe_loader:
                    candidate_active = ~candidate["tr_padding_mask"]
                    candidate_valid = candidate["known_gate_feature_valid"][..., 1]
                    candidate_values = candidate["known_gate_features"][..., 1]
                    if bool(
                        (
                            candidate_active
                            & candidate_valid
                            & (candidate_values != 0.0)
                        ).any()
                    ):
                        gradient_probe = candidate
                        break
                active = ~gradient_probe["tr_padding_mask"]
                drift_valid = gradient_probe["known_gate_feature_valid"][..., 1]
                drift_values = gradient_probe["known_gate_features"][..., 1]
                drift_active = active & drift_valid
                drift_invalid = active & ~drift_valid
                if not bool((drift_active & (drift_values != 0.0)).any()):
                    raise AssertionError(
                        "V24a real training probe contains no nonzero valid RD-drift event"
                    )
                if bool((drift_values.masked_select(drift_invalid) != 0.0).any()):
                    raise AssertionError("V24a invalid RD-drift probe values are not zero")
                model.zero_grad(set_to_none=True)
                gradient_output = model(**_model_inputs(gradient_probe))
                gradient_objective = torch.nn.functional.cross_entropy(
                    gradient_output["window_logits"], gradient_probe["labels"]
                )
                gradient_objective.backward()
                drift_gradient = model.fusion[0].weight.grad[:, -1]
                if (
                    drift_gradient is None
                    or not torch.isfinite(drift_gradient).all()
                    or float(drift_gradient.abs().max().item()) <= 0.0
                ):
                    raise AssertionError(
                        "V24a real training probe gives no gradient to RD-drift column"
                    )
                paired_initialization.update(
                    {
                        "real_training_probe_active_event_count": int(active.sum().item()),
                        "real_training_probe_parent_count": len(
                            gradient_probe["parent_sample_ids"]
                        ),
                        "real_training_probe_valid_rd_drift_event_count": int(
                            drift_active.sum().item()
                        ),
                        "real_training_probe_invalid_rd_drift_event_count": int(
                            drift_invalid.sum().item()
                        ),
                        "real_training_probe_nonzero_valid_rd_drift_event_count": int(
                            (drift_active & (drift_values != 0.0)).sum().item()
                        ),
                        "real_training_probe_rd_drift_gradient_abs_max": float(
                            drift_gradient.abs().max().item()
                        ),
                        "real_training_probe_labels_match_inputs": True,
                        "real_training_probe_rd_drift_column_receives_gradient": True,
                    }
                )
                model.zero_grad(set_to_none=True)
                del gradient_output, gradient_objective, drift_gradient
            del baseline_reference, audit_loader, probe
            del treatment_output, treatment_logits, baseline_logits
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(not parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("V21 must train every model parameter")
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
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
        for epoch in range(start_epoch, config.train.epochs + 1):
            model.train()
            total_loss = 0.0
            final_loss = 0.0
            prefix_loss = 0.0
            gradient_max = 0.0
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
                )
                loss = objective["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item()
                )
                if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
                    raise ValueError("V21 training produced non-finite loss or gradient")
                optimizer.step()
                global_step += 1
                total_loss += float(loss.item())
                final_loss += float(objective["final_state_loss"].item())
                prefix_loss += float(objective["prefix_auxiliary_loss"].item())
                gradient_max = max(gradient_max, grad_norm)
                if step == 1 or step % max(1, min(50, steps_per_epoch // 10)) == 0:
                    logger.info(
                        "progress | variant=%s | epoch=%02d/20 | step=%04d/%04d | total=%.6f | final=%.6f | prefix=%.6f | grad=%.3f",
                        config.variant,
                        epoch,
                        step,
                        steps_per_epoch,
                        float(loss.item()),
                        float(objective["final_state_loss"].item()),
                        float(objective["prefix_auxiliary_loss"].item()),
                        grad_norm,
                    )
            if len(epoch_order) != len(train_dataset) or len(set(epoch_order)) != len(epoch_order):
                raise ValueError("V21 full epoch did not consume each train parent exactly once")
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
                "parent_order_sha256": _parent_order_sha256(epoch_order),
                "parent_order_count": len(epoch_order),
                "parent_order_unique_count": len(set(epoch_order)),
                "elapsed_seconds": time.perf_counter() - epoch_started,
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
        if config.variant == V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
            residual_start = model.base_fusion_input_dim
            residual_end = residual_start + model.rd_temporal_residual_dim
            residual_weights = model.fusion[0].weight.detach()[
                :, residual_start:residual_end
            ]
            window_rows = list(best_validation["records"]["window"])
            multiframe_rows = [
                row for row in window_rows if int(row["rd_frame_count"]) >= 2
            ]
            single_or_missing_rows = [
                row for row in window_rows if int(row["rd_frame_count"]) < 2
            ]
            nonzero_multiframe = sum(
                bool(row["rd_temporal_residual_nonzero"])
                for row in multiframe_rows
            )
            invalid_nonzero = sum(
                bool(row["rd_temporal_residual_nonzero"])
                for row in single_or_missing_rows
            )
            if (
                float(residual_weights.abs().max().item()) <= 0.0
                or not multiframe_rows
                or nonzero_multiframe <= 0
                or invalid_nonzero != 0
            ):
                raise AssertionError(
                    "V22a best checkpoint did not prove valid use of the RD residual"
                )
            rd_temporal_residual_training_audit = {
                "residual_dimension": model.rd_temporal_residual_dim,
                "fusion_columns_zero_initialized": paired_initialization.get(
                    "new_fusion_columns_exactly_zero"
                )
                is True,
                "best_checkpoint_fusion_weight_abs_max": float(
                    residual_weights.abs().max().item()
                ),
                "best_checkpoint_fusion_weight_l2": float(
                    torch.linalg.vector_norm(residual_weights).item()
                ),
                "best_checkpoint_nonzero_fusion_weight_count": int(
                    torch.count_nonzero(residual_weights).item()
                ),
                "validation_multiframe_event_count": len(multiframe_rows),
                "validation_multiframe_nonzero_residual_event_count": int(
                    nonzero_multiframe
                ),
                "validation_single_or_missing_event_count": len(
                    single_or_missing_rows
                ),
                "validation_single_or_missing_nonzero_residual_event_count": int(
                    invalid_nonzero
                ),
                "feature_used_by_best_checkpoint": True,
            }
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one full 20-epoch RADAR-LN3 V21 KnownGate experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    print(
        json.dumps(
            train_known_gate(root, config_path.resolve(), resume_checkpoint=resume),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
