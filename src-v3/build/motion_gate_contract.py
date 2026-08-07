from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from radar_ln3.features import (
    TR_MOTION_CLIP,
    TR_MOTION_MASK_MODE,
    TR_MOTION_NAMES,
    TR_MOTION_SCALES,
    TR_MOTION_SCHEMA,
    TR_MOTION_VALID_NAMES,
)


VERSION = "RADAR-LN3-V9"
SOURCE_CONFIG = "config/RADAR-LN3/RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-config.json"
SOURCE_CONTRACT = "data/processed/RADAR-LN3-V6-cq08-tr-rd-causal-quality-data-contract.json"
SOURCE_INDEX = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv"
SOURCE_STATS = "data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json"
OUTPUT_CONTRACT = "data/processed/RADAR-LN3-V9-cq08-tr-rd-causal-motion-gate-data-contract.json"
OUTPUT_CONFIG = "config/RADAR-LN3/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json"
EXPECTED_HASHES = {
    SOURCE_INDEX: "a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87",
    SOURCE_CONTRACT: "2110938d1d94923edb44c2825e34fb595fb1b34c1353f3f3e512403f91ed8777",
    SOURCE_STATS: "584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6",
}
EXPECTED_SOURCE_CONFIG_SHA256 = "a7c415adc7deeed3b07797d52404e7c36d89e58538d1651dc3b318549a6c71b2"
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
        directory = root / relative
        for path in directory.glob("RADAR-LN3-V*"):
            match = _VERSION_PATTERN.search(path.name)
            if match is not None:
                versions.append(int(match.group("version")))
    return sorted(set(versions))


def _motion_contract() -> dict[str, Any]:
    return {
        "schema": TR_MOTION_SCHEMA,
        "value_dim": len(TR_MOTION_NAMES),
        "value_names": list(TR_MOTION_NAMES),
        "valid_mask_dim": len(TR_MOTION_VALID_NAMES),
        "valid_mask_names": list(TR_MOTION_VALID_NAMES),
        "valid_mask_mode": TR_MOTION_MASK_MODE,
        "scales": dict(TR_MOTION_SCALES),
        "clip_absolute": TR_MOTION_CLIP,
    }


def _build_contract(root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    source = json.loads((root / SOURCE_CONTRACT).read_text(encoding="utf-8"))
    contract = dict(source)
    feature_contract = dict(source["feature_contract"])
    feature_contract["tr_motion"] = _motion_contract()
    feature_contract["missing_policy"] = (
        "TR point missing values retain their original mask; causal TR motion values use a same-shape "
        "boolean valid mask for numerical gating only; invalid and padded motion values are exactly zero; "
        "the motion-valid mask is not concatenated into classifier inputs; RD semantics remain causal-current-event-v1"
    )
    model_io = dict(source["model_io"])
    model_io.update({
        "tr_motion_features": f"[B,L,{len(TR_MOTION_NAMES)}] causal-tr-motion-v1",
        "tr_motion_valid_mask": f"[B,L,{len(TR_MOTION_NAMES)}] bool; gating_only",
        "task_sequence_logits": "[B,L,2] for the V9 DBB/CU Gate only",
        "task_window_logits": "[B,2] for the V9 DBB/CU Gate only",
    })
    contract.update({
        "schema_version": VERSION,
        "contract_name": "cq08_point_level_tr_rd_causal_motion_gate_v1",
        "contract_status": "prepared_for_bounded_validation_not_formally_trained",
        "source_contract": SOURCE_CONTRACT,
        "source_contract_sha256": hashes[SOURCE_CONTRACT],
        "source_index": SOURCE_INDEX,
        "source_index_sha256": hashes[SOURCE_INDEX],
        "source_normalization_stats": SOURCE_STATS,
        "source_normalization_stats_sha256": hashes[SOURCE_STATS],
        "feature_contract": feature_contract,
        "model_io": model_io,
        "training_contract": {
            "tasks": ["dbb_cu_gate"],
            "baseline": "RADAR-LN3-V8 DBB/CU Gate",
            "single_changed_factor": "add causal-tr-motion-v1 values with gating-only valid mask",
            "unchanged": [
                "CQ08 V3 rows, labels, parent grouping and train/val/test split",
                "TR 14-value and 11-missing inputs",
                "RD tensor, RD spatial encoder, causal RD quality and masked_mean frame aggregation",
                "CfC liquid unit, hidden dimensions, causal-prefix loss and class weights",
                "seed, batch size, learning rate, epoch budget, early stopping and parent_min_precision_recall selection",
            ],
            "initialization": "random_model_initialization",
            "pretraining": "not_required_and_not_used",
            "loss_mode": "causal_prefix",
            "checkpoint_selection": "validation parent_min_precision_recall",
            "formal_training_status": "not_executed",
            "test_status": "not_executed",
        },
        "motion_causality": {
            "available_information": "current and earlier points in the same parent only",
            "state_reset_boundary": "every parent/session/target boundary",
            "position_source": "range, azimuth and elevation converted to local radar Cartesian coordinates",
            "future_parent_statistics_forbidden": True,
            "dt_rule": "every interval-dependent feature is invalid when tr_dt_valid=false or dt<=0",
            "epsilon_policy": "no epsilon or fixed interval is used to fabricate a physical derivative",
            "path_rule": "online cumulative path straightness uses only the current causal segment and resets across invalid geometry/time boundaries",
            "angle_rule": "course differences use the shortest wrapped delta in [-180,180) degrees",
            "first_reported_kinematics_rule": "the first event does not seed reported speed/course derivatives because CQ08 initialization semantics are unconfirmed; derivatives begin only after a later observed baseline",
            "consistency_error_rule": "reported/geometric speed and radial/range-rate consistency are encoded as nonnegative absolute errors",
        },
        "leakage_policy": list(source["leakage_policy"]) + [
            "the TR motion-valid mask is used only to zero invalid values and is not a classification feature",
            "no complete-parent duration, dt-valid fraction, endpoint or future path statistic enters V9 tensors",
            "V9 is a closed-set DBB/CU Gate; UnknownTarget classification remains distinct from reject/abstain",
        ],
        "reuse": {
            "index_reuse": True,
            "index_path": SOURCE_INDEX,
            "index_sha256": hashes[SOURCE_INDEX],
            "index_reuse_reason": "Motion values are deterministically derived online from existing causal TR points; data range, labels, split and rows do not change.",
            "normalization_stats_reuse": True,
            "normalization_stats_path": SOURCE_STATS,
            "normalization_stats_sha256": hashes[SOURCE_STATS],
            "normalization_stats_reuse_reason": "RD values and train parent population are unchanged; motion features use frozen physical scales rather than fitted validation/test statistics.",
            "rd_tensors_reused_read_only": True,
            "v8_checkpoint_compatibility": "forbidden; V9 adds a new model/input/checkpoint contract",
        },
        "version_reason": (
            "V9 is the next unused global version after V8 and introduces a substantive Gate input/checkpoint schema change: "
            "nine causal TR motion-consistency values with a same-shape gating-only validity mask. The V3 index, V2 RD statistics and RD tensors are reused unchanged."
        ),
        "validation_status": {
            "contract_file_created": True,
            "focused_motion_tests": "not_executed_at_generation_time",
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
    payload = dict(source)
    payload["schema_version"] = VERSION
    payload["config_name"] = "RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion"
    data = dict(source["data"])
    data.update({
        "contract_path": OUTPUT_CONTRACT,
        "contract_schema_version": VERSION,
        "expected_index_sha256": hashes[SOURCE_INDEX],
        "expected_contract_sha256": contract_hash,
        "expected_stats_sha256": hashes[SOURCE_STATS],
        "tr_motion_schema": TR_MOTION_SCHEMA,
        "tr_motion_mask_mode": TR_MOTION_MASK_MODE,
    })
    payload["data"] = data
    model = dict(source["model"])
    model["tr_motion_dim"] = len(TR_MOTION_NAMES)
    payload["model"] = model
    train = dict(source["train"])
    artifact_dir = "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal"
    train.update({
        "checkpoint_path": f"{artifact_dir}/best_model.pt",
        "last_checkpoint_path": f"{artifact_dir}/last_model.pt",
        "formal_artifact_dir": artifact_dir,
        "checkpoint_schema": "RADAR-LN3-V9-dbb-cu-gate-motion-checkpoint-v1",
    })
    payload["train"] = train
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    tracking_path = root / "src-v3" / "process_v3.md"
    if "Stage 034" not in tracking_path.read_text(encoding="utf-8"):
        raise RuntimeError("Stage 034 must be recorded before V9 generation")
    versions = _existing_versions(root)
    targets_exist = (root / OUTPUT_CONTRACT).exists() and (root / OUTPUT_CONFIG).exists()
    if not versions or (max(versions) != 8 and not (max(versions) >= 9 and targets_exist)):
        raise RuntimeError(f"V9 must follow global V8 without a gap; observed versions={versions}")
    for relative in (SOURCE_CONFIG, *EXPECTED_HASHES):
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    hashes = {relative: sha256_file(root / relative) for relative in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"V9 source hash mismatch: expected={EXPECTED_HASHES}, actual={hashes}")
    source_config_hash = sha256_file(root / SOURCE_CONFIG)
    if source_config_hash != EXPECTED_SOURCE_CONFIG_SHA256:
        raise ValueError(
            "V8 Gate baseline config changed before V9 generation: "
            f"expected={EXPECTED_SOURCE_CONFIG_SHA256}, actual={source_config_hash}"
        )

    contract_path = root / OUTPUT_CONTRACT
    contract = _build_contract(root, hashes)
    contract_status = _write_new_or_identical(contract_path, contract)
    contract_hash = sha256_file(contract_path)
    config_path = root / OUTPUT_CONFIG
    config = _build_config(root, contract_hash, hashes)
    config_status = _write_new_or_identical(config_path, config)
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
        "motion_schema": TR_MOTION_SCHEMA,
        "motion_dim": len(TR_MOTION_NAMES),
        "motion_mask_mode": TR_MOTION_MASK_MODE,
        "formal_training": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the V9 causal TR-motion DBB/CU Gate contract and config.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
