from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    DBB_ANGULAR_EMBEDDING_DIM,
    DBB_ANGULAR_MARGIN,
    DBB_ANGULAR_PAIRED_PROTOCOL,
    DBB_ANGULAR_SCALE,
    DBB_KNOWNNESS_COSINE,
    DBB_KNOWNNESS_PROBABILITY,
    DBB_READOUT_ANGULAR_PROTOTYPE,
    DBB_READOUT_LINEAR,
)


VERSION = ANGULAR_KNOWNNESS_SCHEMA_VERSION
V14_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V14-cq08-strict-open-set-data-contract.json"
)
V15_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V15-cq08-angular-knownness-experiment-contract.json"
)
V14_CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json",
}
TASK_SLUGS = {
    "dbb_cu_gate": "dbb-cu-gate",
    "dbb_specialist": "dbb-specialist",
    "cu_specialist": "cu-specialist",
}
ARMS = ("control", "treatment")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V15 artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _verify_identical(path: Path, payload: dict[str, Any]) -> str:
    """Validate one frozen preparation artifact without touching its timestamp or content."""

    if not path.is_file():
        raise FileNotFoundError(f"V15 preparation artifact is missing: {path}")
    expected = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"V15 preparation artifact differs from the frozen contract: {path}")
    return "verified_identical"


def config_relative(arm: str, task_name: str) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-{arm}-"
        f"{TASK_SLUGS[task_name]}-config.json"
    )


def artifact_relative(arm: str) -> Path:
    return Path(
        "artifacts/"
        f"RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-{arm}-formal"
    )


def _experiment_contract(v14: dict[str, Any], *, v14_hash: str) -> dict[str, Any]:
    payload = copy.deepcopy(v14)
    payload["schema_version"] = VERSION
    payload["contract_name"] = "cq08_v15_strict_open_set_dbb_angular_knownness_paired_v1"
    payload["contract_status"] = "implementation_prepared_formal_training_not_executed"
    payload["source_data_contract"] = _relative(V14_CONTRACT_RELATIVE)
    payload["source_data_contract_sha256"] = v14_hash
    payload["strict_open_set_contract"]["unknown_training_exposure"] = 0
    payload["model_io"].update(
        {
            "dbb_inference_logits": (
                "[B,L,3]; Control linear logits or Treatment fixed-scale cosine logits; "
                "both are label-free at inference"
            ),
            "dbb_knownness_scores": (
                "Control top softmax probability/gap; Treatment raw cosine/top-cosine gap"
            ),
            "dbb_treatment_embedding": (
                f"[B,L,{DBB_ANGULAR_EMBEDDING_DIM}] L2-normalized; DBB known parents only in loss"
            ),
        }
    )
    payload["training_contract"].update(
        {
            "experiment_type": "paired_single_factor_control_treatment",
            "paired_protocol": DBB_ANGULAR_PAIRED_PROTOCOL,
            "control_variable": "linear DBB readout with probability knownness",
            "treatment_variable": (
                "L2-normalized DBB embedding/prototypes with additive cosine target margin"
            ),
            "treatment_embedding_dim": DBB_ANGULAR_EMBEDDING_DIM,
            "treatment_cosine_scale": DBB_ANGULAR_SCALE,
            "treatment_angular_margin": DBB_ANGULAR_MARGIN,
            "margin_scope": "train loss only on known DBB target logits",
            "inference_margin": 0.0,
            "unchanged_factors": [
                "V3 index and split",
                "V14 known-only normalization",
                "TR/RD inputs and masks",
                "liquid hidden dimension and dynamics",
                "Gate and CU architectures",
                "class weights, prefix weight, batch size and epoch budget",
            ],
            "task_specific_initialization": "sha256(paired_seed:task_name) with RNG restoration",
            "unknown_optimizer_parent_count": 0,
        }
    )
    payload["oof_calibration_contract"].update(
        {
            "prediction_protocol": "out_of_fold",
            "epoch_protocol": "nested_fold_train_selection_then_outer_train_retrain",
            "outer_held_out_used_for_normalization": False,
            "outer_held_out_used_for_checkpoint_selection": False,
            "outer_held_out_prediction_pass_count": 1,
            "inner_selection_population": "outer_train_known_parents_only",
            "inner_selection_unknown_count": 0,
            "dbb_control_score": DBB_KNOWNNESS_PROBABILITY,
            "dbb_treatment_score": DBB_KNOWNNESS_COSINE,
            "status": "未执行",
        }
    )
    payload["action_contract"].update(
        {
            "dbb_knownness_score_is_schema_bound": True,
            "control_dbb_knownness_score": DBB_KNOWNNESS_PROBABILITY,
            "treatment_dbb_knownness_score": DBB_KNOWNNESS_COSINE,
            "v14_calibration_compatibility": "forbidden",
        }
    )
    payload["reuse"].update(
        {
            "v14_known_only_normalization_reused_read_only": True,
            "v14_data_contract_reused_as_source": True,
            "v14_checkpoint_compatibility": "forbidden",
            "v14_calibration_compatibility": "forbidden",
            "version_reason": (
                "V15 changes DBB readout/loss/knownness score, checkpoint semantics and OOF epoch "
                "selection while reusing V3 rows/split/TR/RD tensors and V14 known-only normalization"
            ),
        }
    )
    payload["validation_status"].update(
        {
            "focused": "未执行",
            "bounded_cpu_smoke": "未执行",
            "bounded_cuda_smoke": "未执行",
            "formal_training": "未执行",
            "oof_calibration": "未执行",
            "development_unknown_audit": "未执行",
            "test": "未执行",
            "p95": "未执行",
        }
    )
    return payload


def _task_config(
    base: dict[str, Any],
    *,
    arm: str,
    task_name: str,
    contract_hash: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    task_slug = TASK_SLUGS[task_name]
    counterpart = "treatment" if arm == "control" else "control"
    artifact = artifact_relative(arm)
    task_artifact = artifact / task_name
    payload["schema_version"] = VERSION
    payload["config_name"] = (
        f"RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-{arm}-{task_slug}"
    )
    payload["data"].update(
        {
            "contract_path": _relative(V15_CONTRACT_RELATIVE),
            "contract_schema_version": VERSION,
            "expected_contract_sha256": contract_hash,
        }
    )
    model = payload["model"]
    model.update(
        {
            "dbb_readout_mode": DBB_READOUT_LINEAR,
            "dbb_embedding_dim": 0,
            "dbb_cosine_scale": 1.0,
            "dbb_angular_margin": 0.0,
            "dbb_knownness_score_name": DBB_KNOWNNESS_PROBABILITY,
        }
    )
    if arm == "treatment" and task_name == "dbb_specialist":
        model.update(
            {
                "dbb_readout_mode": DBB_READOUT_ANGULAR_PROTOTYPE,
                "dbb_embedding_dim": DBB_ANGULAR_EMBEDDING_DIM,
                "dbb_cosine_scale": DBB_ANGULAR_SCALE,
                "dbb_angular_margin": DBB_ANGULAR_MARGIN,
                "dbb_knownness_score_name": DBB_KNOWNNESS_COSINE,
            }
        )
    train = payload["train"]
    train.update(
        {
            "checkpoint_path": _relative(task_artifact / "best_model.pt"),
            "last_checkpoint_path": _relative(task_artifact / "last_model.pt"),
            "formal_artifact_dir": _relative(task_artifact),
            "epoch_checkpoint_dir": _relative(task_artifact / "epochs"),
            "joint_artifact_dir": _relative(artifact),
            "joint_best_checkpoint_path": _relative(artifact / "joint_best_model.pt"),
            "joint_last_checkpoint_path": _relative(artifact / "joint_last_model.pt"),
            "joint_checkpoint_schema": f"{VERSION}-angular-knownness-{arm}-joint-checkpoint-v1",
            "checkpoint_schema": f"{VERSION}-angular-knownness-{arm}-{task_slug}-checkpoint-v1",
            "paired_protocol": DBB_ANGULAR_PAIRED_PROTOCOL,
            "paired_arm": arm,
            "paired_seed": int(train["seed"]),
            "paired_counterpart_config_path": _relative(
                config_relative(counterpart, task_name)
            ),
        }
    )
    return payload


def build(project_root: Path, *, verify_only: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / "config/RADAR-LN3"
    v14_contract_path = root / V14_CONTRACT_RELATIVE
    if not v14_contract_path.is_file():
        raise FileNotFoundError(v14_contract_path)
    v14_contract_hash = sha256_file(v14_contract_path)
    if v14_contract_hash != "062b67f7c9b33b682a55ce90a6b41e5d66bafada374e149ae46c02392a68f378":
        raise ValueError("frozen V14 strict-open-set data contract hash changed")
    v14_contract = json.loads(v14_contract_path.read_text(encoding="utf-8"))
    if v14_contract.get("schema_version") != "RADAR-LN3-V14":
        raise ValueError("V15 source data contract is not the frozen V14 contract")
    contract_payload = _experiment_contract(v14_contract, v14_hash=v14_contract_hash)
    contract_path = root / V15_CONTRACT_RELATIVE
    contract_write_status = (
        _verify_identical(contract_path, contract_payload)
        if verify_only
        else _write_new_or_identical(contract_path, contract_payload)
    )
    contract_hash = sha256_file(contract_path)

    base_configs: dict[str, dict[str, Any]] = {}
    expected_v14_hashes = {
        "dbb_cu_gate": "9d1d708cc3955c1584357a3ea5207b1076d9ad3c59cd8b99d89648c7b0faa8b3",
        "dbb_specialist": "23d2f158b368003b3b1c927c6cd7b14cd10d86e09d5c1d431135d571e2ac2da0",
        "cu_specialist": "3beddc09d60a47edf15276d2be92b8345d0a9fb2e99fdcf1c9643f99252e3af6",
    }
    for task_name, file_name in V14_CONFIG_NAMES.items():
        path = config_root / file_name
        if not path.is_file() or sha256_file(path) != expected_v14_hashes[task_name]:
            raise ValueError(f"frozen V14 base config changed: {path}")
        base_configs[task_name] = json.loads(path.read_text(encoding="utf-8"))

    outputs: dict[str, Any] = {}
    for arm in ARMS:
        outputs[arm] = {}
        if (
            not verify_only
            and (root / artifact_relative(arm)).exists()
            and any((root / artifact_relative(arm)).iterdir())
        ):
            raise FileExistsError(f"refusing to prepare over a non-empty V15 artifact: {root / artifact_relative(arm)}")
        for task_name in V14_CONFIG_NAMES:
            payload = _task_config(
                base_configs[task_name],
                arm=arm,
                task_name=task_name,
                contract_hash=contract_hash,
            )
            path = root / config_relative(arm, task_name)
            outputs[arm][task_name] = {
                "path": str(path),
                "write_status": (
                    _verify_identical(path, payload)
                    if verify_only
                    else _write_new_or_identical(path, payload)
                ),
                "sha256": sha256_file(path),
            }
    return {
        "status": "passed",
        "mode": "verify_only" if verify_only else "prepare",
        "schema_version": VERSION,
        "experiment": "paired DBB linear Control versus angular-prototype Treatment",
        "contract_path": str(contract_path),
        "contract_sha256": contract_hash,
        "contract_write_status": contract_write_status,
        "source_v14_contract_sha256": v14_contract_hash,
        "configs": outputs,
        "index_split_changed": False,
        "normalization_changed": False,
        "unknown_training_exposure": 0,
        "formal_training": "未执行",
        "nested_oof": "未执行",
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V15 paired DBB angular-knownness experiment without training."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read and verify frozen V15 preparation artifacts without creating or overwriting files.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root.resolve(), verify_only=bool(args.verify_only)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
