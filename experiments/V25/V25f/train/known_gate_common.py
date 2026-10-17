from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


from radar_ln3.known_gate import KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA, KNOWN_GATE_BASELINE_MULTISEED_SEEDS, KNOWN_GATE_CLASS_NAMES, KNOWN_GATE_CONTRACT_SCHEMA, KNOWN_GATE_MANIFEST_SCHEMA, KNOWN_GATE_READINESS_SCHEMA, KNOWN_GATE_V21E_MANIFEST_SCHEMA, KNOWN_GATE_V24_EXPERIMENT_SCHEMA, KNOWN_GATE_V24_MANIFEST_SCHEMA, KNOWN_GATE_V24A_EXPERIMENT_SCHEMA, KNOWN_GATE_V24A_MANIFEST_SCHEMA, KNOWN_GATE_V24C_EXPERIMENT_SCHEMA, KNOWN_GATE_V24C_MANIFEST_SCHEMA, KNOWN_GATE_V25A_EXPERIMENT_SCHEMA, KNOWN_GATE_V25A_MANIFEST_SCHEMA, KNOWN_GATE_V25A_READINESS_SCHEMA, KNOWN_GATE_V25D_EXPERIMENT_SCHEMA, KNOWN_GATE_V25D_MANIFEST_SCHEMA, KNOWN_GATE_V25E_EXPERIMENT_SCHEMA, KNOWN_GATE_V25E_MANIFEST_SCHEMA, KNOWN_GATE_V26_EXPERIMENT_SCHEMA, KNOWN_GATE_V26_MANIFEST_SCHEMA, UNKNOWN_LABEL, KnownGateConfig, sha256_file
from radar_ln3.known_gate_features import BASELINE_VARIANT, COURSE_PREFIX_FEATURE_SCHEMA, V21A_INTERFRAME_VARIANT, V21B_VARIANT, V21C_VARIANT, V21D_PREFIX_VARIANT, V21E_COURSE_PREFIX_VARIANT, V22A_RD_TEMPORAL_RESIDUAL_VARIANT, V24_COURSE_REASON_AWARE_BASELINE_VARIANT, V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT, V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT, V25A_GRADIENT_TELEMETRY_VARIANT, V25D_PRESENT_CLASS_MEAN_VARIANT, V25E_SHARED_GRADIENT_PROJECTION_VARIANT, V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT


READINESS_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V21-known-gate-training-readiness.json"
)
MULTISEED_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json"
)
V25A_READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-training-readiness.json"
)




def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()










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



















