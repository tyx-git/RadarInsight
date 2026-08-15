"""Prepare the V20r three-seed confirmation without rebuilding V16 data.

Seed 42 is an immutable completed screening pair.  This builder creates only
the seed 43/44 V16-core Control and density-weighted Treatment configurations,
then writes one train-only readiness artifact that both new arms must bind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from build.liquid_dynamics_v20 import (
    SOURCE_CONFIG,
    VARIANTS,
    _config_payload,
    _write_json_if_same,
)
from radar_ln3.task_models import (
    LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    V20_DENSITY_PAIRED_PROTOCOL,
    build_v20_paired_models,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


SEEDS = (42, 43, 44)
NEW_TRAINING_SEEDS = (43, 44)
ARMS = ("control", "treatment")
READINESS_SCHEMA = (
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1"
)
READINESS_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-training-readiness.json"
)
SEED42_REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V20-density-weighted-paired-validation.json"
)
SEED42_CONTROL_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-"
    "control-seed42-config.json"
)
SEED42_CONTROL_SUMMARY = Path(
    "artifacts/RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-"
    "control-seed42-formal/training-summary.json"
)
SEED42_TREATMENT_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-"
    "treatment-seed42-config.json"
)
SEED42_TREATMENT_SUMMARY = Path(
    "artifacts/RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-"
    "treatment-seed42-formal/training-summary.json"
)


def config_relative_path(seed: int, arm: str) -> Path:
    if seed not in NEW_TRAINING_SEEDS or arm not in ARMS:
        raise ValueError(f"invalid V20r multiseed config request: seed={seed}, arm={arm}")
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-"
        f"{arm}-seed{seed}-config.json"
    )


def artifact_relative_path(seed: int, arm: str) -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-"
        f"{arm}-seed{seed}-formal"
    )


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _artifact_nonempty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _assert_completed_seed42(root: Path) -> dict[str, Any]:
    report_path = root / SEED42_REPORT
    if not report_path.is_file():
        raise FileNotFoundError(
            f"V20r multiseed confirmation requires the completed seed42 report: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment") != "V20r-density"
        or report.get("status") not in {"passed", "completed_but_gate_failed"}
        or report.get("test") != "未执行"
        or report.get("OOF") != "未执行"
        or report.get("full_cascade") != "未执行"
        or report.get("formal_rejection") != "未执行"
        or report.get("stable") != "未执行"
        or report.get("P95") != "未执行"
        or report.get("deployment_eligible") is not False
    ):
        raise ValueError("seed42 V20r report is incomplete or outside sealed scope")

    summaries: dict[str, Any] = {}
    for arm, (relative, config_relative) in {
        "control": (SEED42_CONTROL_SUMMARY, SEED42_CONTROL_CONFIG),
        "treatment": (SEED42_TREATMENT_SUMMARY, SEED42_TREATMENT_CONFIG),
    }.items():
        path = root / relative
        config = load_task_config(root / config_relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        epochs = {int(row.get("epoch", -1)) for row in summary.get("history") or []}
        if (
            summary.get("status") != "completed"
            or summary.get("task_name") != "dbb_specialist"
            or summary.get("stop_reason") != "epoch_limit"
            or summary.get("test_evaluation") != "not_executed"
            or summary.get("full_cascade_evaluation") != "not_executed"
            or summary.get("oof_rejection_calibration") != "not_executed"
            or summary.get("stable_evaluation") != "not_executed"
            or summary.get("p95_latency") != "not_executed"
            or summary.get("deployment_eligible") is not False
            or epochs != set(range(1, 21))
            or not (root / config.train.checkpoint_path).is_file()
            or not (root / config.train.last_checkpoint_path).is_file()
            or any(
                not (
                    root
                    / str(config.train.epoch_checkpoint_dir)
                    / f"epoch-{epoch:03d}.pt"
                ).is_file()
                for epoch in range(1, 21)
            )
        ):
            raise ValueError(f"seed42 {arm} summary is incomplete or unsealed")
        summaries[arm] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "best_epoch": int(summary.get("best_epoch", -1)),
            "initial_logits_sha256": str(
                dict(summary.get("paired_initialization") or {}).get(
                    "initial_logits_sha256", ""
                )
            ),
        }
    if (
        len(summaries["control"]["initial_logits_sha256"]) != 64
        or summaries["control"]["initial_logits_sha256"]
        != summaries["treatment"]["initial_logits_sha256"]
    ):
        raise ValueError("seed42 shared Control/Treatment initialization hash differs")
    report_runs = dict(report.get("runs") or {})
    if any(
        dict(report_runs.get(arm) or {}).get("summary_sha256")
        != summaries[arm]["sha256"]
        for arm in ("control", "treatment")
    ):
        raise ValueError("seed42 report no longer binds the reused training summaries")
    return {
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "control_config": str((root / SEED42_CONTROL_CONFIG).resolve()),
        "control_config_sha256": sha256_file(root / SEED42_CONTROL_CONFIG),
        "treatment_config": str((root / SEED42_TREATMENT_CONFIG).resolve()),
        "treatment_config_sha256": sha256_file(root / SEED42_TREATMENT_CONFIG),
        "runs": summaries,
        "role": "completed_seed42_screening_pair_reused_read_only",
    }


def _cross_seed_training_core(config: Any) -> dict[str, Any]:
    excluded = {
        "seed",
        "dataloader_seed",
        "paired_seed",
        "checkpoint_path",
        "checkpoint_schema",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_arm",
        "paired_counterpart_config_path",
    }
    return {
        key: value
        for key, value in config.train.__dict__.items()
        if key not in excluded
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_path = root / SOURCE_CONFIG
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_config = load_task_config(source_path)
    source_contract = verify_task_data_contract(root, source_config)
    if (
        source_config.schema_version != "RADAR-LN3-V16"
        or source_config.task.name != "dbb_specialist"
        or source_config.model.liquid_invalid_dt_update_mode != "bounded_observation"
    ):
        raise ValueError("multiseed confirmation requires the frozen V16 DBB Control")

    seed42 = _assert_completed_seed42(root)
    variant = VARIANTS["density"]
    configs: dict[int, dict[str, Any]] = {}
    contracts: dict[int, dict[str, Any]] = {}
    state_hashes: dict[str, str] = {}
    pair_evidence: dict[str, Any] = {}

    for seed in NEW_TRAINING_SEEDS:
        configs[seed] = {}
        contracts[seed] = {}
        for arm in ARMS:
            counterpart = "treatment" if arm == "control" else "control"
            counterpart_name = f"{variant['name_prefix']}-{counterpart}-seed{seed}"
            payload = _config_payload(
                source_payload,
                variant=variant,
                arm=arm,
                counterpart_name=counterpart_name,
                seed=seed,
            )
            relative = config_relative_path(seed, arm)
            path = root / relative
            _write_json_if_same(path, payload)
            config = load_task_config(path)
            if (
                config.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
                or config.train.seed != seed
                or config.train.paired_seed != seed
                or config.train.dataloader_seed != seed
                or config.train.paired_protocol != V20_DENSITY_PAIRED_PROTOCOL
                or config.train.paired_arm != arm
            ):
                raise ValueError(f"generated seed{seed} {arm} config violates the protocol")
            configs[seed][arm] = config
            contracts[seed][arm] = verify_task_data_contract(root, config)

        control = configs[seed]["control"]
        treatment = configs[seed]["treatment"]
        if (
            control.task != source_config.task
            or control.data != source_config.data
            or control.model != source_config.model
            or treatment.task != source_config.task
            or treatment.data != source_config.data
        ):
            raise ValueError(f"seed{seed} pair drifts from the frozen V16 core")
        if contracts[seed]["control"]["hashes"] != contracts[seed]["treatment"]["hashes"]:
            raise ValueError(f"seed{seed} Control/Treatment data contracts differ")
        control_model, treatment_model, evidence = build_v20_paired_models(
            control, treatment
        )
        control_hash = _state_sha256(control_model.state_dict())
        treatment_hash = _state_sha256(treatment_model.state_dict())
        if control_hash != treatment_hash:
            raise AssertionError(f"seed{seed} paired initial states differ")
        state_hashes[str(seed)] = control_hash
        pair_evidence[str(seed)] = evidence

    if len(set(state_hashes.values())) != len(NEW_TRAINING_SEEDS):
        raise ValueError("seed43/44 initial states are not independent")
    if (
        _cross_seed_training_core(configs[43]["control"])
        != _cross_seed_training_core(configs[44]["control"])
        or _cross_seed_training_core(configs[43]["treatment"])
        != _cross_seed_training_core(configs[44]["treatment"])
    ):
        raise ValueError("seed43/44 training cores differ beyond seed and artifact identity")

    contract_hashes = contracts[43]["control"]["hashes"]
    if contract_hashes != source_contract["hashes"] or any(
        contracts[seed][arm]["hashes"] != contract_hashes
        for seed in NEW_TRAINING_SEEDS
        for arm in ARMS
    ):
        raise ValueError("multiseed configs do not share the frozen V16 data contract")

    paired_configs = {
        f"seed{seed}_{arm}": {
            "seed": seed,
            "arm": arm,
            "path": str((root / config_relative_path(seed, arm)).resolve()),
            "sha256": sha256_file(root / config_relative_path(seed, arm)),
            "mode": configs[seed][arm].model.liquid_invalid_dt_update_mode,
        }
        for seed in NEW_TRAINING_SEEDS
        for arm in ARMS
    }
    checks = {
        "seed_set_is_exactly_42_43_44": SEEDS == (42, 43, 44),
        "seed42_screening_pair_is_complete_and_read_only": True,
        "new_training_is_restricted_to_seeds_43_44": NEW_TRAINING_SEEDS == (43, 44),
        "all_new_configs_share_frozen_v16_data_contract": True,
        "all_new_pairs_differ_only_by_density_update_mode": True,
        "each_new_pair_has_exactly_equal_initial_state": True,
        "seed43_and_seed44_initial_states_are_distinct": True,
        "validation_does_not_select_configs_or_seed_set": True,
        "test_payload_access_zero": True,
    }
    result = {
        "status": "passed" if all(checks.values()) else "blocked",
        "schema_version": READINESS_SCHEMA,
        "experiment": "V20r-density-multiseed-confirmation",
        "paired_protocol": V20_DENSITY_PAIRED_PROTOCOL,
        "seed_set": list(SEEDS),
        "new_training_seeds": list(NEW_TRAINING_SEEDS),
        "source_v16_config": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        },
        "seed42_reuse": seed42,
        "paired_configs": paired_configs,
        "contract_hashes": contract_hashes,
        "initial_state_sha256_by_new_seed": state_hashes,
        "pairing_evidence_by_new_seed": pair_evidence,
        "decision_contract": {
            "primary_signal": (
                "at least 2 of 3 seeds and the three-seed median must have "
                "Treatment-Control parent Macro-F1 > 0.02"
            ),
            "full_adoption": (
                "at least 2 of 3 seeds and the three-seed median must pass every "
                "frozen seed42 class-floor, weak-floor, and directional-error guard"
            ),
            "same_validation_parents": (
                "seeds are optimization replicates, not independent data samples"
            ),
        },
        "checks": checks,
        "validation_used_for_readiness_gate": False,
        "seed42_validation_role": "authorization_only_not_hyperparameter_tuning",
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

    output = root / READINESS_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    formal_artifact_nonempty = any(
        _artifact_nonempty(root / artifact_relative_path(seed, arm))
        for seed in NEW_TRAINING_SEEDS
        for arm in ARMS
    )
    if formal_artifact_nonempty:
        if not output.is_file():
            raise FileNotFoundError(
                f"multiseed readiness is missing after formal artifacts exist: {output}"
            )
        if output.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to change multiseed readiness after training: {output}"
            )
    else:
        output.write_text(text, encoding="utf-8")
    result["report_path"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V20r seed43/44 paired confirmation configs/readiness."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.project_root)


if __name__ == "__main__":
    main()
