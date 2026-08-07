from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from radar_ln3.features import (
    TR_OBSERVATION_INCREMENT_SCHEMA,
    resolve_causal_tr_feature_contract,
)


VERSION = "RADAR-LN3-V10"
SOURCE_CONFIG = "config/RADAR-LN3/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json"
SOURCE_CONTRACT = "data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json"
SOURCE_INDEX = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv"
SOURCE_STATS = "data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json"
OUTPUT_CONTRACT = "data/processed/RADAR-LN3-V10-cq08-tr-rd-observation-increment-gate-data-contract.json"
OUTPUT_CONFIG = "config/RADAR-LN3/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json"
ARTIFACT_DIR = "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal"
EXPECTED_HASHES = {
    SOURCE_INDEX: "a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87",
    SOURCE_CONTRACT: "2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777",
    SOURCE_STATS: "584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6",
}
EXPECTED_SOURCE_CONFIG_SHA256 = "ac2722b810a4557a6e72ddfbbd956393783b0ca1c4154281d01f78af93fda7a3"
_VERSION_PATTERN = re.compile(r"RADAR-LN3-V(?P<version>\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different versioned file: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _existing_versions(root: Path) -> list[int]:
    versions: list[int] = []
    for relative in ("data/processed", "config/RADAR-LN3"):
        for path in (root / relative).glob("RADAR-LN3-V*"):
            match = _VERSION_PATTERN.search(path.name)
            if match is not None:
                versions.append(int(match.group("version")))
    return sorted(set(versions))


def _build_contract(root: Path, hashes: dict[str, str], source_config_hash: str) -> dict[str, Any]:
    source = json.loads((root / SOURCE_CONTRACT).read_text(encoding="utf-8"))
    increment = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
    contract = dict(source)
    feature_contract = dict(source["feature_contract"])
    feature_contract["tr_motion"] = increment.as_dict()
    feature_contract["missing_policy"] = (
        "TR point missing values retain their original mask; causal observation-increment values use a same-shape "
        "boolean valid mask for numerical gating only; invalid and padded values are exactly zero; the valid mask "
        "is not concatenated into classifier inputs; values and validity never consume dt, dt_valid or timestamps; "
        "RD semantics remain causal-current-event-v1"
    )
    model_io = dict(source["model_io"])
    model_io.update({
        "tr_motion_features": f"[B,L,{increment.value_dim}] {increment.schema}",
        "tr_motion_valid_mask": f"[B,L,{increment.value_dim}] bool; gating_only",
        "task_sequence_logits": "[B,L,2] for the V10 DBB/CU Gate only",
        "task_window_logits": "[B,2] for the V10 DBB/CU Gate only",
    })
    time_contract = dict(source["time_contract"])
    time_contract.update({
        "derived_tr_feature_dt_dependency": False,
        "derived_tr_feature_dt_valid_dependency": False,
        "liquid_dynamics_time_contract_unchanged": True,
    })
    contract.update({
        "schema_version": VERSION,
        "contract_name": "cq08_point_level_tr_rd_observation_increment_gate_v1",
        "contract_status": "prepared_for_bounded_validation_not_formally_trained",
        "source_contract": SOURCE_CONTRACT,
        "source_contract_sha256": hashes[SOURCE_CONTRACT],
        "source_index": SOURCE_INDEX,
        "source_index_sha256": hashes[SOURCE_INDEX],
        "source_normalization_stats": SOURCE_STATS,
        "source_normalization_stats_sha256": hashes[SOURCE_STATS],
        "paired_baseline_config": SOURCE_CONFIG,
        "paired_baseline_config_sha256": source_config_hash,
        "time_contract": time_contract,
        "feature_contract": feature_contract,
        "model_io": model_io,
        "training_contract": {
            "tasks": ["dbb_cu_gate"],
            "paired_baseline": "RADAR-LN3-V9 DBB/CU Gate",
            "single_changed_factor": (
                "replace causal-tr-motion-v1 dt-rate values with causal-tr-observation-increment-v1 values; "
                "the nine-dimensional encoder architecture is unchanged"
            ),
            "unchanged": [
                "CQ08 V3 rows, labels, parent grouping and train/val/test split",
                "TR 14-value and 11-missing inputs",
                "RD tensor, RD spatial encoder, causal RD quality and masked_mean frame aggregation",
                "CfC liquid unit and its existing dt/dt_valid dynamics, hidden dimensions and classifier",
                "causal-prefix loss, class weights, seed, batch size and learning rate",
                "epoch budget, early stopping and parent_min_precision_recall checkpoint selection",
            ],
            "initialization": "random_model_initialization; V9 weights are not reused",
            "pretraining": "not_required_and_not_used",
            "loss_mode": "causal_prefix",
            "checkpoint_selection": "validation parent_min_precision_recall",
            "formal_training_status": "not_executed",
            "test_status": "not_executed",
        },
        "observation_increment_causality": {
            "available_information": "current and earlier TR observations in the same parent only",
            "state_reset_boundary": "every parent/session/target boundary",
            "time_inputs_forbidden": ["tr_time_seconds", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason"],
            "future_parent_statistics_forbidden": True,
            "position_source": "current/previous range, azimuth and elevation converted to local radar Cartesian coordinates",
            "angle_rule": "course changes use the shortest wrapped delta in [-180,180) degrees",
            "first_event_rule": "no predecessor, so every increment value is invalid and exactly zero",
            "first_reported_kinematics_rule": (
                "first-event speed/course do not seed the next delta because CQ08 initialization-zero semantics remain unconfirmed"
            ),
            "units_warning": (
                "values describe change per observation/report and are not SI rates per second; report-cadence domain shift remains a validation risk"
            ),
        },
        "leakage_policy": list(source["leakage_policy"]) + [
            "V10 observation-increment values and validity never read dt, dt_valid, timestamp precision or tr_dt_reason",
            "the same-shape validity mask gates invalid values only and is not a classifier input",
            "no complete-parent duration, endpoint, dt-valid fraction or future path statistic enters V10 tensors",
            "V10 is a closed-set DBB/CU Gate; UnknownTarget classification remains distinct from reject/abstain",
        ],
        "reuse": {
            "index_reuse": True,
            "index_path": SOURCE_INDEX,
            "index_sha256": hashes[SOURCE_INDEX],
            "index_reuse_reason": (
                "Observation increments are derived online from existing causal TR points; data range, labels, split and rows do not change."
            ),
            "normalization_stats_reuse": True,
            "normalization_stats_path": SOURCE_STATS,
            "normalization_stats_sha256": hashes[SOURCE_STATS],
            "normalization_stats_reuse_reason": (
                "RD values and train parents are unchanged; increments use frozen physical scales rather than fitted validation/test statistics."
            ),
            "rd_tensors_reused_read_only": True,
            "v9_checkpoint_compatibility": "forbidden; identical tensor dimension does not imply identical feature semantics",
        },
        "version_reason": (
            "V10 is the next unused global version after V9 and changes the Gate feature/checkpoint contract from dt-dependent "
            "physical-rate motion to no-dt per-observation increments. The V3 index, split, RD tensors and V2 RD statistics are unchanged."
        ),
        "validation_status": {
            "contract_file_created": True,
            "focused_increment_tests": "not_executed_at_generation_time",
            "bounded_smoke": "not_executed_at_generation_time",
            "formal_training": "not_executed",
            "validation_performance": "not_evaluated",
            "test": "not_executed",
            "multi_seed": "not_executed",
            "p95": "not_executed",
        },
    })
    return contract


def _build_config(root: Path, contract_hash: str, hashes: dict[str, str]) -> dict[str, Any]:
    source = json.loads((root / SOURCE_CONFIG).read_text(encoding="utf-8"))
    increment = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
    payload = dict(source)
    payload["schema_version"] = VERSION
    payload["config_name"] = "RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment"
    data = dict(source["data"])
    data.update({
        "contract_path": OUTPUT_CONTRACT,
        "contract_schema_version": VERSION,
        "expected_index_sha256": hashes[SOURCE_INDEX],
        "expected_contract_sha256": contract_hash,
        "expected_stats_sha256": hashes[SOURCE_STATS],
        "tr_motion_schema": increment.schema,
        "tr_motion_mask_mode": increment.mask_mode,
    })
    payload["data"] = data
    model = dict(source["model"])
    model["tr_motion_dim"] = increment.value_dim
    payload["model"] = model
    train = dict(source["train"])
    train.update({
        "checkpoint_path": f"{ARTIFACT_DIR}/best_model.pt",
        "last_checkpoint_path": f"{ARTIFACT_DIR}/last_model.pt",
        "formal_artifact_dir": ARTIFACT_DIR,
        "checkpoint_schema": "RADAR-LN3-V10-dbb-cu-gate-observation-increment-checkpoint-v1",
    })
    payload["train"] = train
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    process_path = root / "src-v3" / "process_v3.md"
    if "Stage 036" not in process_path.read_text(encoding="utf-8"):
        raise RuntimeError("Stage 036 must be recorded before V10 generation")
    versions = _existing_versions(root)
    contract_path = root / OUTPUT_CONTRACT
    config_path = root / OUTPUT_CONFIG
    targets_exist = contract_path.exists() and config_path.exists()
    if not versions or (max(versions) != 9 and not (max(versions) >= 10 and targets_exist)):
        raise RuntimeError(f"V10 must follow global V9 without a gap; observed versions={versions}")
    for relative in (SOURCE_CONFIG, *EXPECTED_HASHES):
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    artifact_path = root / ARTIFACT_DIR
    if artifact_path.exists() and not targets_exist:
        raise FileExistsError(f"V10 formal artifact exists before its frozen config/contract: {artifact_path}")
    hashes = {relative: sha256_file(root / relative) for relative in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"V10 source hash mismatch: expected={EXPECTED_HASHES}, actual={hashes}")
    source_config_hash = sha256_file(root / SOURCE_CONFIG)
    if source_config_hash != EXPECTED_SOURCE_CONFIG_SHA256:
        raise ValueError(
            "V9 Gate paired baseline config changed before V10 generation: "
            f"expected={EXPECTED_SOURCE_CONFIG_SHA256}, actual={source_config_hash}"
        )

    contract = _build_contract(root, hashes, source_config_hash)
    contract_status = _write_new_or_identical(contract_path, contract)
    contract_hash = sha256_file(contract_path)
    config = _build_config(root, contract_hash, hashes)
    config_status = _write_new_or_identical(config_path, config)
    increment = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
    return {
        "status": "generated_or_reused_identical",
        "schema_version": VERSION,
        "contract_path": OUTPUT_CONTRACT,
        "contract_status": contract_status,
        "contract_sha256": contract_hash,
        "config_path": OUTPUT_CONFIG,
        "config_status": config_status,
        "config_sha256": sha256_file(config_path),
        "reused_index": SOURCE_INDEX,
        "index_sha256": hashes[SOURCE_INDEX],
        "reused_stats": SOURCE_STATS,
        "stats_sha256": hashes[SOURCE_STATS],
        "feature_schema": increment.schema,
        "feature_dim": increment.value_dim,
        "feature_mask_mode": increment.mask_mode,
        "formal_artifact_dir": ARTIFACT_DIR,
        "formal_training": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the V10 no-dt observation-increment DBB/CU Gate contract and config.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
