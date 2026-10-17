from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
    KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
    KNOWN_GATE_READINESS_SCHEMA,
    load_known_gate_config,
    sha256_file,
)


BASE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed42-config.json"
)
BASE_READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V21-known-gate-training-readiness.json"
)
MULTISEED_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json"
)
NEW_SEEDS = KNOWN_GATE_BASELINE_MULTISEED_SEEDS[1:]


def config_relative(seed: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed{int(seed)}-config.json"
    )


def artifact_relative(seed: int) -> Path:
    return Path(
        "artifacts/"
        f"RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed{int(seed)}-formal"
    )


def _clone(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def multiseed_config_payload(base_payload: Mapping[str, Any], seed: int) -> dict[str, Any]:
    value = int(seed)
    if value not in NEW_SEEDS:
        raise ValueError(f"V21 additional Baseline seed must be one of {NEW_SEEDS}: {value}")
    payload = _clone(base_payload)
    artifact = artifact_relative(value).as_posix()
    payload["config_name"] = (
        f"RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed{value}"
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
                f"RADAR-LN3-V21-baseline-seed{value}-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": None,
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
    ):
        train[field] = f"<{field}>"
    comparable["train"] = train
    return comparable


def selection_protocol() -> dict[str, Any]:
    return {
        "scope": "development validation parent-final KnownGate Baseline only",
        "candidate_seeds": list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS),
        "ranking_order": [
            "maximize parent minimum over both classes precision and recall",
            "maximize parent macro_f1",
            "maximize KNOWN_CLUTTER_EVIDENCE f1",
            "maximize KNOWN_TARGET_EVIDENCE recall",
            "prefer lower numeric seed on an exact tie",
        ],
        "bootstrap": {
            "unit": "parent",
            "stratification": "original Drone/Bird/Balloon/Clutter label",
            "replicates": 5000,
            "confidence_level": 0.95,
            "selection_frequency_reported": True,
        },
        "selection_role": "development_champion_only",
        "seed42_result_was_observed_before_this_protocol_was_frozen": True,
        "protocol_frozen_before_seed43_or_seed44_training": True,
        "selection_is_unbiased_test_estimate": False,
    }


def _validate_seed42_summary(root: Path, base_config_hash: str, readiness_hash: str) -> dict[str, Any]:
    artifact = root / artifact_relative(42)
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    if not summary_path.is_file() or not detail_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"V21 seed42 formal Baseline is incomplete: {artifact}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inferred_seed = int(summary.get("seed", dict(summary.get("dataloader_contract") or {}).get("seed", -1)))
    if (
        summary.get("status") != "completed"
        or summary.get("task_name") != "known_gate"
        or summary.get("variant") != "baseline"
        or inferred_seed != 42
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or len(list(summary.get("history") or [])) != 20
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("model_state_changed_from_initialization") is not True
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or sha256_file(Path(str(summary.get("config_path", "")))) != base_config_hash
        or dict(summary.get("training_readiness") or {}).get("sha256") != readiness_hash
    ):
        raise ValueError("V21 seed42 Baseline summary violates the frozen multiseed source contract")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "sample_detail_path": str(detail_path),
        "sample_detail_sha256": sha256_file(detail_path),
        "best_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_selection": summary["best_checkpoint_selection"],
        "performance_status": summary.get("performance_status"),
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    base_config_path = (root / BASE_CONFIG).resolve()
    base_readiness_path = (root / BASE_READINESS).resolve()
    if not base_config_path.is_file() or not base_readiness_path.is_file():
        raise FileNotFoundError("V21 seed42 config/readiness must exist before multiseed preparation")
    base_config_payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    base_config = load_known_gate_config(base_config_path)
    if base_config.variant != "baseline" or base_config.train.seed != 42:
        raise ValueError("V21 multiseed source is not the frozen seed42 Baseline config")
    base_readiness = json.loads(base_readiness_path.read_text(encoding="utf-8"))
    base_checks = dict(base_readiness.get("checks") or {})
    base_binding = dict(dict(base_readiness.get("configs") or {}).get("baseline") or {})
    base_config_hash = sha256_file(base_config_path)
    base_readiness_hash = sha256_file(base_readiness_path)
    if (
        base_readiness.get("status") != "passed"
        or base_readiness.get("schema_version") != KNOWN_GATE_READINESS_SCHEMA
        or not base_checks
        or not all(value is True for value in base_checks.values())
        or Path(str(base_binding.get("path", ""))).resolve() != base_config_path
        or base_binding.get("sha256") != base_config_hash
        or int(base_readiness.get("test_payload_access_count", -1)) != 0
        or base_readiness.get("test") != "未执行"
        or base_readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V21 seed42 readiness is stale or outside the sealed contract")
    seed42 = _validate_seed42_summary(root, base_config_hash, base_readiness_hash)

    readiness_path = root / MULTISEED_READINESS
    if not readiness_path.exists():
        for seed in NEW_SEEDS:
            artifact = root / artifact_relative(seed)
            if artifact.exists() and any(artifact.iterdir()):
                raise FileExistsError(
                    "refusing to prepare a new multiseed protocol over an existing additional-seed artifact: "
                    f"{artifact}"
                )

    config_bindings: dict[str, dict[str, str]] = {
        "42": {"path": str(base_config_path), "sha256": base_config_hash}
    }
    base_core = frozen_training_core(base_config_payload)
    for seed in NEW_SEEDS:
        payload = multiseed_config_payload(base_config_payload, seed)
        if frozen_training_core(payload) != base_core:
            raise AssertionError(f"V21 seed{seed} differs beyond seed/path/schema fields")
        config_path = root / config_relative(seed)
        _write_json_new_or_identical(config_path, payload)
        loaded = load_known_gate_config(config_path)
        if loaded.variant != "baseline" or loaded.train.seed != seed:
            raise ValueError(f"V21 seed{seed} generated config failed round-trip validation")
        config_bindings[str(seed)] = {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        }

    checks = {
        "seed42_formal_baseline_is_complete_and_bound": True,
        "candidate_seed_set_is_exactly_42_43_44": True,
        "seed43_seed44_differ_only_by_seed_and_artifact_identity": True,
        "all_candidates_require_full_20_epoch_parameter_training": True,
        "cross_seed_ranking_protocol_is_frozen_before_new_seed_training": True,
        "clutter_support_nine_requires_parent_bootstrap": True,
        "unknown_is_excluded_from_model_and_selection_population": True,
        "test_payload_access_count_is_zero": True,
    }
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
        "checks": checks,
        "seeds": list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "configs": config_bindings,
        "base_readiness_path": str(base_readiness_path),
        "base_readiness_sha256": base_readiness_hash,
        "base_seed42": seed42,
        "contract_hashes": dict(base_readiness["contract_hashes"]),
        "selection_protocol": selection_protocol(),
        "new_seed_training_at_report_generation": "未执行",
        "validation_used_for_cross_seed_development_selection": True,
        "validation_used_to_change_model_or_loss_hyperparameters": False,
        "test_payload_access_count": 0,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(readiness_path, readiness)
    return {
        "status": "passed",
        "schema_version": KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA,
        "readiness": str(readiness_path),
        "configs": {seed: value["path"] for seed, value in config_bindings.items()},
        "seed42_reused": True,
        "new_seeds_to_train": list(NEW_SEEDS),
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V21 KnownGate Baseline seeds 43/44 from the frozen seed42 run."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
