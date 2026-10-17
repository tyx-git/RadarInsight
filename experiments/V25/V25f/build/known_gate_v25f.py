"""Prepare the V25f frozen causal-prefix checkpoint-selection scan."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

from build.known_gate_v25a import (
    ARTIFACT as V25A_ARTIFACT,
    CONFIG as V25A_CONFIG,
    EXPERIMENT_CONTRACT as V25A_EXPERIMENT_CONTRACT,
    FEATURE_MANIFEST as V25A_FEATURE_MANIFEST,
    READINESS as V25A_READINESS,
)
from radar_ln3.known_gate import (
    build_known_gate_model_for_config,
    load_known_gate_config,
    known_gate_config_identity_sha256,
    sha256_file,
)
from radar_ln3.known_gate_v25f_selection import (
    V25F_DIAGNOSTIC_MAX_KNOWN_TARGET_PARENT_LOSS,
    V25F_MINIMUM_CLUTTER_PARENT_SUPPORT,
    V25F_MINIMUM_CLUTTER_PRECISION,
    V25F_MINIMUM_CLUTTER_RECALL,
    V25F_MINIMUM_CLUTTER_TRUE_POSITIVES,
    V25F_PREFIX_MINIMUM_EVENTS,
    V25F_SELECTOR_SCHEMA,
)
from train.known_gate_common import _verify_contract, _verify_readiness


EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25f-cq08-known-gate-causal-prefix-"
    "checkpoint-utility-selection-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection.json"
)
EXPECTED_EPOCHS = tuple(range(1, 21))
EXPECTED_TRAIN_PARENTS = 986
EXPECTED_VALIDATION_PARENTS = 232
EXPECTED_VALIDATION_CLUTTER = 9
MAX_V25F_JSON_BYTES = 64 * 1024 * 1024
MAX_V25F_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_V25F_CHECKPOINT_MEMBERS = 512

CONTRACT_SCHEMA = (
    "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-"
    "utility-selection-experiment-contract-v1"
)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.stat().st_size > MAX_V25F_JSON_BYTES:
            raise ValueError(
                "existing V25f artifact exceeds the fail-closed JSON size limit: "
                f"{path}={path.stat().st_size} > {MAX_V25F_JSON_BYTES} bytes"
            )
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different V25f artifact: {path}")
        return
    if len(content.encode("utf-8")) > MAX_V25F_JSON_BYTES:
        raise ValueError(
            "new V25f artifact exceeds the fail-closed JSON size limit: "
            f"{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        # A failed write must not leave a file that can be mistaken for a
        # complete V25f contract on the next invocation.
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V25f JSON source is not a file: {path}")
    size = path.stat().st_size
    if size > MAX_V25F_JSON_BYTES:
        raise ValueError(
            "V25f JSON source exceeds the fail-closed size limit: "
            f"{path}={size} > {MAX_V25F_JSON_BYTES} bytes"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"V25f JSON source must contain an object: {path}")
    return dict(payload)


def validate_v25f_checkpoint_file(path: Path) -> int:
    """Validate the bounded archive envelope before invoking torch.load."""

    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError(f"V25f epoch checkpoint must be a .pt file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_V25F_CHECKPOINT_BYTES:
        raise ValueError(
            "V25f epoch checkpoint exceeds the fail-closed size limit: "
            f"{path}={size} > {MAX_V25F_CHECKPOINT_BYTES} bytes"
        )
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_V25F_CHECKPOINT_MEMBERS:
                raise ValueError(
                    "V25f epoch checkpoint contains an invalid number of ZIP members: "
                    f"{path}={len(members)}"
                )
            declared_uncompressed = 0
            for member in members:
                if member.is_dir() or member.file_size < 0:
                    raise ValueError(
                        f"V25f epoch checkpoint contains an invalid ZIP member: {path}"
                    )
                declared_uncompressed += int(member.file_size)
                if declared_uncompressed > MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "V25f epoch checkpoint declared uncompressed size exceeds "
                        f"the fail-closed limit: {path}"
                    )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"V25f epoch checkpoint is not a valid ZIP .pt archive: {path}") from exc
    return size


def _source(root: Path) -> dict[str, Any]:
    paths = {
        "config": root / V25A_CONFIG,
        "readiness": root / V25A_READINESS,
        "feature_manifest": root / V25A_FEATURE_MANIFEST,
        "feature_contract": root / V25A_EXPERIMENT_CONTRACT,
        "summary": root / V25A_ARTIFACT / "training-summary.json",
        "sample_detail": root / V25A_ARTIFACT / "sample_detail_information.json",
        "artifact": root / V25A_ARTIFACT,
        "experiment_document": root
        / "src-v3/docs/radar_ln3_v25a_preclip_gradient_telemetry_experiment.md",
    }
    missing = [
        name
        for name, path in paths.items()
        if (
            not path.is_dir()
            if name == "artifact"
            else not path.is_file()
        )
    ]
    if missing:
        raise FileNotFoundError(f"V25f frozen V25a sources are missing: {missing}")
    config = load_known_gate_config(paths["config"])
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, paths["config"], config, contract)
    summary = _load_json(paths["summary"])
    detail = _load_json(paths["sample_detail"])
    history = list(summary.get("history") or ())
    orders = list(dict(summary.get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or ())
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != "v25a_course_preclip_gradient_telemetry"
        or int(summary.get("seed", -1)) != 43
        or int(summary.get("train_parent_count", -1)) != EXPECTED_TRAIN_PARENTS
        or int(summary.get("validation_parent_count", -1)) != EXPECTED_VALIDATION_PARENTS
        or len(history) != 20
        or [int(row.get("epoch", -1)) for row in history] != list(EXPECTED_EPOCHS)
        or len(orders) != 20
        or [int(row.get("epoch", -1)) for row in orders] != list(EXPECTED_EPOCHS)
        or any(int(row.get("count", -1)) != EXPECTED_TRAIN_PARENTS for row in orders)
        or any(not str(row.get("sha256", "")) for row in orders)
        or int(summary.get("global_step", -1)) != 9860
        or int(summary.get("best_epoch", -1)) != 14
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or detail.get("variant") != summary.get("variant")
        or int(detail.get("epoch", -1)) != 14
    ):
        raise ValueError("V25f frozen V25a training source is stale or incomplete")
    epoch_paths = {
        epoch: paths["artifact"] / "epochs" / f"epoch-{epoch:03d}.pt"
        for epoch in EXPECTED_EPOCHS
    }
    missing_epochs = [epoch for epoch, path in epoch_paths.items() if not path.is_file()]
    if missing_epochs:
        raise FileNotFoundError(f"V25f V25a epoch checkpoints are missing: {missing_epochs}")
    for path in epoch_paths.values():
        validate_v25f_checkpoint_file(path)
    model, _, pairing = build_known_gate_model_for_config(root, config)
    del model
    if (
        config.variant != "v25a_course_preclip_gradient_telemetry"
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V25f source is not the frozen V25a telemetry-only pair")
    return {
        "paths": paths,
        "config": config,
        "contract": contract,
        "readiness": readiness,
        "summary": summary,
        "detail": detail,
        "history": history,
        "orders": {str(row["epoch"]): str(row["sha256"]) for row in orders},
        "epoch_paths": epoch_paths,
        "hashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = _source(root)
    paths = source["paths"]
    hashes = source["hashes"]
    contract = {
        "schema_version": (
            CONTRACT_SCHEMA
        ),
        "selector_schema": V25F_SELECTOR_SCHEMA,
        "experiment_name": "V25f causal-prefix checkpoint utility selection",
        "scope": "development-only KnownGate checkpoint scan; no retraining or product action claim",
        "single_factor": {
            "changed_component": "checkpoint selection rule only",
            "training_step_count": 0,
            "optimizer_step_count": 0,
            "frozen": [
                "V25a self-developed causal-course model",
                "V25a loss and prefix auxiliary weight",
                "V25a features and bounded_observation dt semantics",
                "V25a initialization, DataLoader order and saved 20 epochs",
                "V25a optimizer and gradient clipping",
            ],
        },
        "source_v25a": {
            "config": _relative(root, paths["config"]),
            "config_sha256": hashes["config"],
            "config_identity_sha256": known_gate_config_identity_sha256(source["config"]),
            "readiness": _relative(root, paths["readiness"]),
            "readiness_sha256": hashes["readiness"],
            "feature_manifest": _relative(root, paths["feature_manifest"]),
            "feature_manifest_sha256": hashes["feature_manifest"],
            "feature_contract": _relative(root, paths["feature_contract"]),
            "feature_contract_sha256": hashes["feature_contract"],
            "training_summary": _relative(root, paths["summary"]),
            "training_summary_sha256": hashes["summary"],
            "sample_detail": _relative(root, paths["sample_detail"]),
            "sample_detail_sha256": hashes["sample_detail"],
            "artifact": _relative(root, paths["artifact"]),
            "legacy_best_epoch": 14,
            "epoch_checkpoint_count": 20,
            "epoch_parent_order_sha256": source["orders"],
        },
        "threshold_policy": {
            "product_safe": {
                "clutter_parent_support": V25F_MINIMUM_CLUTTER_PARENT_SUPPORT,
                "clutter_true_positive_minimum": V25F_MINIMUM_CLUTTER_TRUE_POSITIVES,
                "clutter_recall_minimum": V25F_MINIMUM_CLUTTER_RECALL,
                "clutter_precision_minimum": V25F_MINIMUM_CLUTTER_PRECISION,
                "known_target_parent_recall": "must_not_decrease_vs_V25a_legacy",
                "known_target_prefix_recall": "must_not_decrease_vs_V25a_legacy",
            },
            "diagnostic_frontier": {
                "purpose": "locate the selection tradeoff only; never an adoption gate",
                "maximum_known_target_parent_loss": V25F_DIAGNOSTIC_MAX_KNOWN_TARGET_PARENT_LOSS,
            },
            "fixed_delta_gate": "not_used; effect is reported with paired parent bootstrap and exact parent counts",
        },
        "prefix_action_semantics": {
            "minimum_evidence_events": V25F_PREFIX_MINIMUM_EVENTS,
            "events_before_minimum": "PENDING_ABSTAIN",
            "eligible_prefix_excludes_final_event": True,
            "prefix_loss_target": "full parent-final KnownGate label",
            "local_argmax_is_product_action": False,
            "unknown_probability_output": False,
        },
        "evaluation": {
            "validation_parent_count": EXPECTED_VALIDATION_PARENTS,
            "validation_clutter_parent_support": EXPECTED_VALIDATION_CLUTTER,
            "primary": "parent-balanced eligible-prefix Clutter F1 after product-safe filtering",
            "required": [
                "all 20 epoch candidates",
                "strict and diagnostic feasible epoch counts",
                "parent-final and causal-prefix P/R/F1",
                "pending/eligible action counts",
                "stratified paired parent bootstrap",
                "per-parent action trajectories",
            ],
        },
        "training": "not_executed_by_contract",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    readiness = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection-readiness-v1",
        "selector_schema": V25F_SELECTOR_SCHEMA,
        "experiment_contract": {
            "path": _relative(root, root / EXPERIMENT_CONTRACT),
            "sha256": contract_hash,
        },
        "source_v25a": {
            "config_path": _relative(root, paths["config"]),
            "config_sha256": hashes["config"],
            "readiness_path": _relative(root, paths["readiness"]),
            "readiness_sha256": hashes["readiness"],
            "summary_path": _relative(root, paths["summary"]),
            "summary_sha256": hashes["summary"],
            "legacy_best_epoch": 14,
            "epoch_checkpoint_count": 20,
        },
        "checks": {
            "source_v25a_training_completed": True,
            "source_v25a_has_all_twenty_epoch_checkpoints": True,
            "source_v25a_parent_orders_are_sealed": True,
            "source_v25a_test_not_executed": True,
            "source_v25a_unknown_training_exposure_is_zero": True,
            "no_v25f_training_or_optimizer_steps": True,
            "product_safe_and_diagnostic_thresholds_are_distinct": True,
            "fixed_two_pp_gate_is_not_used": True,
        },
        "training": "not_executed",
        "validation_used_for_epoch_scan": True,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "full_cascade": "not_executed",
        "formal_rejection": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": "passed",
        "schema_version": readiness["schema_version"],
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "readiness": str(root / READINESS),
        "source_v25a_summary": str(paths["summary"]),
        "legacy_best_epoch": 14,
        "epoch_checkpoint_count": 20,
        "training": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare V25f checkpoint utility selection.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()