"""Prepare the V21e seed42/43/44 paired confirmation.

The three completed V21 Baseline runs are immutable controls.  Seed43 V21e is
the completed screening Treatment.  This builder creates only the seed42 and
seed44 Treatment configs and one frozen readiness artifact; it never rebuilds
the V16/V21 data or feature caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.known_gate_v21_multiseed import (
    artifact_relative as baseline_artifact_relative,
    config_relative as baseline_config_relative,
)
from build.known_gate_v21e import (
    CONFIG as SEED43_TREATMENT_CONFIG,
    EXPERIMENT_CONTRACT,
    FEATURE_MANIFEST,
    READINESS as SEED43_TREATMENT_READINESS,
    TRAIN_CACHE,
    VALIDATION_CACHE,
    _clone,
    _state_dict_sha256,
)
from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
    KNOWN_GATE_V21E_MULTISEED_SEEDS,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import V21E_COURSE_PREFIX_VARIANT


SEEDS = KNOWN_GATE_V21E_MULTISEED_SEEDS
NEW_TRAINING_SEEDS = (42, 44)
BASELINE_MULTISEED_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json"
)
BASELINE_MULTISEED_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-validation.json"
)
SEED43_TREATMENT_ARTIFACT = Path(
    "artifacts/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal"
)
SEED43_PAIRED_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-paired-validation.json"
)
PROTOCOL_DOCUMENT = Path(
    "docs/radar_ln3_v21e_multiseed_confirmation.md"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
    "multiseed-training-readiness.json"
)


def treatment_config_relative(seed: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
        f"seed{int(seed)}-config.json"
    )


def treatment_artifact_relative(seed: int) -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
        f"seed{int(seed)}-formal"
    )


def multiseed_config_payload(source: Mapping[str, Any], seed: int) -> dict[str, Any]:
    value = int(seed)
    if value not in NEW_TRAINING_SEEDS:
        raise ValueError(
            f"V21e new Treatment seed must be one of {NEW_TRAINING_SEEDS}: {value}"
        )
    payload = _clone(source)
    artifact = treatment_artifact_relative(value).as_posix()
    payload["config_name"] = (
        "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
        f"seed{value}"
    )
    train = dict(payload["train"])
    train.update(
        {
            "seed": value,
            "dataloader_seed": value,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V21-v21e_course_prefix-"
                f"seed{value}-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": baseline_config_relative(value).as_posix(),
        }
    )
    payload["train"] = train
    return payload


def frozen_training_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparable = _clone(payload)
    comparable["config_name"] = "<seed-bound-config-name>"
    train = dict(comparable["train"])
    for field in (
        "seed",
        "dataloader_seed",
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
        "paired_baseline_config_path",
    ):
        train[field] = f"<{field}>"
    comparable["train"] = train
    return comparable


def _artifact_nonempty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _completed_run(
    root: Path,
    *,
    config_relative: Path,
    artifact_relative: Path,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    config_path = root / config_relative
    artifact = root / artifact_relative
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    last_path = artifact / "last_model.pt"
    if not all(path.is_file() for path in (config_path, summary_path, checkpoint_path, last_path)):
        raise FileNotFoundError(f"V21e multiseed source run is incomplete: {artifact}")
    config = load_known_gate_config(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inferred_seed = int(
        summary.get(
            "seed",
            dict(summary.get("dataloader_contract") or {}).get("seed", -1),
        )
    )
    epochs = {int(row.get("epoch", -1)) for row in summary.get("history") or []}
    if (
        config.variant != variant
        or config.train.seed != seed
        or summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != variant
        or inferred_seed != seed
        or epochs != set(range(1, 21))
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("oof_threshold_calibration") != "not_executed"
        or summary.get("full_cascade_evaluation") != "not_executed"
        or summary.get("formal_rejection") != "not_executed"
        or summary.get("stable_evaluation") != "not_executed"
        or summary.get("p95_latency") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or Path(str(summary.get("config_path", ""))).resolve() != config_path.resolve()
        or Path(str(summary.get("best_checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or any(
            not (artifact / "epochs" / f"epoch-{epoch:03d}.pt").is_file()
            for epoch in range(1, 21)
        )
    ):
        raise ValueError(f"V21e multiseed source run violates the frozen contract: {artifact}")
    return {
        "seed": seed,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_model_state_sha256": str(summary.get("initial_model_state_sha256", "")),
        "best_epoch": int(summary.get("best_epoch", -1)),
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    required = {
        "source_treatment_config": root / SEED43_TREATMENT_CONFIG,
        "source_treatment_readiness": root / SEED43_TREATMENT_READINESS,
        "source_treatment_report": root / SEED43_PAIRED_REPORT,
        "baseline_multiseed_readiness": root / BASELINE_MULTISEED_READINESS,
        "baseline_multiseed_report": root / BASELINE_MULTISEED_REPORT,
        "experiment_contract": root / EXPERIMENT_CONTRACT,
        "feature_manifest": root / FEATURE_MANIFEST,
        "train_cache": root / TRAIN_CACHE,
        "validation_cache": root / VALIDATION_CACHE,
        "protocol_document": root / PROTOCOL_DOCUMENT,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V21e multiseed sources are missing: {missing}")

    source_payload = json.loads(
        required["source_treatment_config"].read_text(encoding="utf-8")
    )
    source_config = load_known_gate_config(required["source_treatment_config"])
    source_readiness = json.loads(
        required["source_treatment_readiness"].read_text(encoding="utf-8")
    )
    source_report = json.loads(
        required["source_treatment_report"].read_text(encoding="utf-8")
    )
    baseline_readiness = json.loads(
        required["baseline_multiseed_readiness"].read_text(encoding="utf-8")
    )
    baseline_report = json.loads(
        required["baseline_multiseed_report"].read_text(encoding="utf-8")
    )
    if (
        source_config.variant != V21E_COURSE_PREFIX_VARIANT
        or source_config.train.seed != 43
        or source_readiness.get("status") != "passed"
        or source_report.get("status") != "completed"
        or dict(source_report.get("frozen_decision") or {}).get(
            "robust_positive_signal"
        )
        is not True
        or dict(source_report.get("frozen_decision") or {}).get(
            "advance_only_to_multiseed_or_train_parent_grouped_oof"
        )
        is not True
        or baseline_readiness.get("status") != "passed"
        or tuple(baseline_readiness.get("seeds") or ()) != SEEDS
        or int(baseline_report.get("selected_seed", -1)) != 43
        or baseline_report.get("test") != "未执行"
        or source_report.get("test") != "未执行"
        or source_report.get("deployment_eligible") is not False
    ):
        raise ValueError("V21e seed43 authorization or V21 Baseline multiseed source is stale")

    baseline_runs: dict[str, Any] = {}
    baseline_initial_hashes: dict[int, str] = {}
    for seed in SEEDS:
        run = _completed_run(
            root,
            config_relative=baseline_config_relative(seed),
            artifact_relative=baseline_artifact_relative(seed),
            variant="baseline",
            seed=seed,
        )
        baseline_config = load_known_gate_config(root / baseline_config_relative(seed))
        rebuilt, reference, evidence = build_known_gate_model_for_config(
            root, baseline_config
        )
        rebuilt_hash = _state_dict_sha256(rebuilt.state_dict())
        if (
            reference is not None
            or evidence.get("paired_baseline") is not False
            or rebuilt_hash != run["initial_model_state_sha256"]
        ):
            raise ValueError(f"current code cannot reconstruct Baseline seed{seed}")
        baseline_initial_hashes[seed] = rebuilt_hash
        baseline_runs[str(seed)] = run
        del rebuilt

    seed43_run = _completed_run(
        root,
        config_relative=SEED43_TREATMENT_CONFIG,
        artifact_relative=SEED43_TREATMENT_ARTIFACT,
        variant=V21E_COURSE_PREFIX_VARIANT,
        seed=43,
    )
    source_treatment_binding = dict(source_report.get("treatment") or {})
    if (
        source_treatment_binding.get("summary_sha256") != seed43_run["summary_sha256"]
        or source_treatment_binding.get("checkpoint_sha256")
        != seed43_run["checkpoint_sha256"]
    ):
        raise ValueError("V21e seed43 report no longer binds the completed Treatment")

    if any(
        _artifact_nonempty(root / treatment_artifact_relative(seed))
        for seed in NEW_TRAINING_SEEDS
    ) and not (root / READINESS).is_file():
        raise FileNotFoundError(
            "V21e multiseed readiness is missing after new-seed artifacts exist"
        )

    treatment_configs: dict[str, Any] = {}
    pairing_evidence: dict[str, Any] = {}
    generated_payloads: dict[int, dict[str, Any]] = {}
    for seed in NEW_TRAINING_SEEDS:
        payload = multiseed_config_payload(source_payload, seed)
        generated_payloads[seed] = payload
        path = root / treatment_config_relative(seed)
        _write_json_new_or_identical(path, payload)
        config = load_known_gate_config(path)
        treatment, baseline_reference, evidence = build_known_gate_model_for_config(
            root, config
        )
        if (
            config.variant != V21E_COURSE_PREFIX_VARIANT
            or config.train.seed != seed
            or baseline_reference is None
            or evidence.get("shared_parameters_and_buffers_exactly_equal") is not True
            or evidence.get("new_fusion_columns_exactly_zero") is not True
            or int(evidence.get("new_fusion_columns", -1)) != 1
            or _state_dict_sha256(baseline_reference.state_dict())
            != baseline_initial_hashes[seed]
        ):
            raise ValueError(f"generated V21e seed{seed} pair violates initialization")
        treatment_configs[str(seed)] = {
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "paired_baseline_config_path": str(
                (root / baseline_config_relative(seed)).resolve()
            ),
        }
        pairing_evidence[str(seed)] = evidence
        del treatment, baseline_reference

    source_core = frozen_training_core(source_payload)
    if any(frozen_training_core(payload) != source_core for payload in generated_payloads.values()):
        raise ValueError("V21e multiseed config drifts beyond seed/artifact identity")
    if len(set(baseline_initial_hashes.values())) != len(SEEDS):
        raise ValueError("V21e Baseline initial states are not distinct across seeds")

    source_contract_hashes = dict(source_readiness.get("contract_hashes") or {})
    checks = {
        "seed_set_is_exactly_42_43_44": SEEDS == (42, 43, 44),
        "new_training_is_restricted_to_seed42_and_seed44_treatments": (
            NEW_TRAINING_SEEDS == (42, 44)
        ),
        "all_three_completed_baselines_are_full_20_epoch_controls": True,
        "seed43_completed_treatment_is_sha_bound_and_reused_read_only": True,
        "seed43_screening_result_authorizes_multiseed_only": True,
        "seed42_and_seed44_treatments_share_the_frozen_v21e_feature_contract": True,
        "each_new_treatment_copies_its_same_seed_baseline_initialization": True,
        "each_new_treatment_zeroes_exactly_one_new_fusion_column": True,
        "seed42_43_44_initial_states_are_distinct": True,
        "validation_does_not_select_the_new_seed_configs": True,
        "unknown_remains_excluded_and_test_payload_access_is_zero": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "blocked",
        "schema_version": KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
        "experiment": "V21e-causal-course-multiseed-confirmation",
        "seed_set": list(SEEDS),
        "new_training_seeds": list(NEW_TRAINING_SEEDS),
        "checks": checks,
        "protocol_document": {
            "path": str(required["protocol_document"].resolve()),
            "sha256": sha256_file(required["protocol_document"]),
        },
        "baseline_multiseed_source": {
            "readiness_path": str(required["baseline_multiseed_readiness"].resolve()),
            "readiness_sha256": sha256_file(required["baseline_multiseed_readiness"]),
            "report_path": str(required["baseline_multiseed_report"].resolve()),
            "report_sha256": sha256_file(required["baseline_multiseed_report"]),
        },
        "baseline_runs": baseline_runs,
        "treatment_configs": treatment_configs,
        "pairing_evidence_by_new_seed": pairing_evidence,
        "seed43_reuse": {
            **seed43_run,
            "readiness_path": str(required["source_treatment_readiness"].resolve()),
            "readiness_sha256": sha256_file(required["source_treatment_readiness"]),
            "paired_report_path": str(required["source_treatment_report"].resolve()),
            "paired_report_sha256": sha256_file(required["source_treatment_report"]),
            "role": "completed_seed43_screening_treatment_reused_read_only",
        },
        "contract_hashes": source_contract_hashes,
        "post_hoc_feature_discovery": True,
        "decision_contract": {
            "parent_final_replication": (
                "at least 2 of 3 seeds and the three-seed median must have "
                "Treatment-Baseline parent Macro-F1 > 0.02"
            ),
            "direction_safety": (
                "at least 2 of 3 seeds must have positive combined parent net rescue, "
                "no negative base-label net rescue, and non-decreasing KnownTarget/Clutter recall"
            ),
            "online_prefix_guard": (
                "reported separately; it cannot be replaced by parent-final success"
            ),
            "seed_role": (
                "optimization replicates over the same development parents, not independent data samples"
            ),
        },
        "validation_used_for_readiness_gate": False,
        "seed43_validation_role": "candidate_authorization_only",
        "test_payload_access_count": 0,
        "formal_training_at_report_generation": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / READINESS, readiness)
    result = {
        "status": readiness["status"],
        "schema_version": readiness["schema_version"],
        "readiness": str((root / READINESS).resolve()),
        "new_treatment_configs": treatment_configs,
        "reused_seed43_treatment": seed43_run,
        "reused_baseline_seeds": list(SEEDS),
        "test": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V21e seed42/43/44 paired confirmation metadata."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.project_root)


if __name__ == "__main__":
    main()
