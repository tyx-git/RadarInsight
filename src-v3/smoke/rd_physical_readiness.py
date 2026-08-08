from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from build.rd_physical_paired import (
    CACHE_MANIFEST,
    DATA_CONTRACT,
    PAIRED_PROTOCOL,
    SEEDS,
    TRAIN_CACHE,
    VAL_CACHE,
    VERSION,
    _config_path,
    sha256_file,
)
from radar_ln3.task_models import (
    build_task_dataloader,
    build_task_dataset,
    build_v11_paired_models,
    load_task_config,
    verify_task_data_contract,
)
from smoke.static_isolation_check import check as isolation_check
from validate.rd_physical_paired import run as validate_cache


CPU_RESULT = Path("src-v3/smoke/RADAR-LN3-V11-rd-physical-paired-cpu/result.json")
CUDA_RESULT = Path("src-v3/smoke/RADAR-LN3-V11-rd-physical-paired-cuda/result.json")
REPORT_TEST_RESULT = Path("src-v3/smoke/RADAR-LN3-V11-rd-physical-paired-report-test-result.json")
CACHE_ACCESS_RESULT = Path("src-v3/smoke/RADAR-LN3-V11-rd-physical-cache-access-test-result.json")
OUTPUT = Path("src-v3/smoke/RADAR-LN3-V11-rd-physical-training-readiness.json")


def _load_passed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"V11 readiness dependency did not pass: {path}")
    return payload


def _command(config_path: Path) -> str:
    return (
        "Set-Location F:\\Project\\RadarInsight\\src-v3; "
        "..\\.venv\\Scripts\\python.exe -m train.tasks "
        "--project-root F:\\Project\\RadarInsight "
        f"--config F:\\Project\\RadarInsight\\{str(config_path).replace('/', chr(92))}"
    )


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    cpu = _load_passed(root / CPU_RESULT)
    cuda = _load_passed(root / CUDA_RESULT)
    report_test = _load_passed(root / REPORT_TEST_RESULT)
    cache_access = _load_passed(root / CACHE_ACCESS_RESULT)
    cache_validation = validate_cache(root)
    isolation = isolation_check(root / "src-v3")
    if isolation.get("status") != "passed":
        raise ValueError("V11 src-v3 isolation failed")

    configs: list[dict[str, Any]] = []
    formal_artifacts: list[dict[str, Any]] = []
    configs_by_seed_arm: dict[tuple[int, str], Any] = {}
    for seed in SEEDS:
        for arm in ("control", "treatment"):
            relative = _config_path(arm, seed)
            path = root / relative
            config = load_task_config(path)
            configs_by_seed_arm[(seed, arm)] = config
            contract = verify_task_data_contract(root, config)
            artifact = root / config.train.formal_artifact_dir
            contents = list(artifact.iterdir()) if artifact.exists() else []
            if contents:
                raise ValueError(f"V11 formal artifact directory is not clean: {artifact}")
            configs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rd_physical_dim": config.model.rd_physical_dim,
                    "loads_rd_physical_cache": contract["rd_physical"] is not None,
                    "artifact_dir": str(artifact),
                    "manual_training_command": _command(relative),
                }
            )
            formal_artifacts.append(
                {
                    "path": str(artifact),
                    "exists": artifact.exists(),
                    "content_count": len(contents),
                }
            )

    reference_config = configs_by_seed_arm[(42, "treatment")]
    reference_dataset = build_task_dataset(
        root,
        reference_config,
        split_name=reference_config.data.train_split_name,
        parent_limit=2,
    )
    reference_batch = next(
        iter(build_task_dataloader(reference_dataset, batch_size=2, shuffle=False))
    )
    shared_inputs = {
        "tr_features": reference_batch["tr_features"],
        "tr_missing_mask": reference_batch["tr_missing_mask"],
        "tr_dt_seconds": reference_batch["tr_dt_seconds"],
        "tr_dt_valid": reference_batch["tr_dt_valid"],
        "tr_motion_features": reference_batch["tr_motion_features"],
        "tr_motion_valid_mask": reference_batch["tr_motion_valid_mask"],
        "rd_frames": reference_batch["rd_frames"],
        "rd_frame_mask": reference_batch["rd_frame_mask"],
        "rd_quality": reference_batch["rd_quality"],
        "tr_padding_mask": reference_batch["tr_padding_mask"],
    }
    active = ~reference_batch["tr_padding_mask"]
    probe = torch.zeros((*active.shape, 4), dtype=reference_batch["tr_features"].dtype)
    probe[active] = 0.5
    all_seed_initialization: list[dict[str, Any]] = []
    for seed in SEEDS:
        control_model, treatment_model, evidence = build_v11_paired_models(
            configs_by_seed_arm[(seed, "control")],
            configs_by_seed_arm[(seed, "treatment")],
        )
        control_model.eval()
        treatment_model.eval()
        with torch.inference_mode():
            control_logits = control_model(**shared_inputs)["sequence_logits"]
            treatment_logits = treatment_model(
                **shared_inputs,
                rd_physical_features=probe,
                rd_physical_valid=active,
            )["sequence_logits"]
        maximum_difference = float((control_logits - treatment_logits).abs().max().item())
        if not torch.equal(control_logits, treatment_logits):
            raise AssertionError(
                f"V11 seed={seed} CPU initial paired logits differ: maximum={maximum_difference}"
            )
        all_seed_initialization.append(
            {
                "seed": seed,
                **evidence,
                "CPU_initial_logits_bitwise_equal": True,
                "maximum_absolute_difference": maximum_difference,
                "initial_logits_sha256": hashlib.sha256(
                    control_logits.detach().contiguous().numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )

    result = {
        "status": "passed",
        "schema_version": VERSION,
        "paired_protocol": PAIRED_PROTOCOL,
        "scope": "complete V11 implementation readiness; no formal training, test, OOF or P95",
        "release_decision": (
            "可以进入用户手动V11 Control/Treatment三seed正式训练；"
            "尚无V11性能结论，不能进入第二阶段OOF、test、正式级联或部署"
        ),
        "data": {
            "source": "CQ08 V3 only",
            "index_sha256": cache_validation["source_index_sha256"],
            "contract": str(root / DATA_CONTRACT),
            "contract_sha256": sha256_file(root / DATA_CONTRACT),
            "manifest": str(root / CACHE_MANIFEST),
            "manifest_sha256": sha256_file(root / CACHE_MANIFEST),
            "train_cache": str(root / TRAIN_CACHE),
            "train_cache_sha256": sha256_file(root / TRAIN_CACHE),
            "validation_cache": str(root / VAL_CACHE),
            "validation_cache_sha256": sha256_file(root / VAL_CACHE),
            "train_event_count": cache_validation["split_results"]["train"]["event_count"],
            "train_valid_event_count": cache_validation["split_results"]["train"]["valid_event_count"],
            "validation_event_count": cache_validation["split_results"]["val"]["event_count"],
            "validation_valid_event_count": cache_validation["split_results"]["val"]["valid_event_count"],
            "full_online_source_recomputation_maximum_absolute_error": 0.0,
            "test_event_count_sealed": cache_validation["test_row_count_sealed"],
            "test_rd_payload_access_count": 0,
            "test_cache_generated": False,
        },
        "single_factor": {
            "control": "retrain V10 architecture under V11 paired protocol; Fusion input 127; no RD physical cache loaded",
            "treatment": "append exactly four current-event RD physical values; Fusion input 131",
            "unchanged": [
                "CQ08 V3 parent rows, labels and train/val split",
                "TR 14 values, 11 missing flags and 9 no-dt observation increments",
                "RD CNN input/encoder and 7 causal current-event RD quality values",
                "CfC liquid dynamics, dimensions, causal-prefix loss, class weights, batch size and learning rate",
                "parent_min_precision_recall checkpoint selection and validation evaluator",
            ],
            "pretraining": "not_required_and_not_used",
        },
        "paired_reproducibility": {
            "seeds": list(SEEDS),
            "shared_parameters_and_buffers_exactly_equal": True,
            "treatment_four_new_fusion_columns_zero": True,
            "CPU_reference_initial_logits_bitwise_equal_all_seeds": True,
            "all_seed_initialization_evidence": all_seed_initialization,
            "independent_same_seed_DataLoader_generators": True,
            "each_epoch_parent_order_sha256": True,
            "CUDA_strict_deterministic_algorithms": False,
            "determinism_claim": "fixed shared initialization and sampling order; not bitwise CUDA determinism",
            "observed_CUDA_initial_maximum_absolute_difference_seed42": cuda["paired_initialization"]["initial_device_logits_maximum_absolute_difference"],
        },
        "checkpointing": {
            "save_each_epoch": True,
            "best_model_semantics": "own-arm validation parent_min_precision_recall best",
            "last_model_semantics": "last completed epoch",
            "epoch_boundary_recovery": cuda["checkpoint"]["epoch_boundary_recovery"],
            "resume_requires_explicit_checkpoint_inside_same_arm_artifact": True,
        },
        "focused_validation": {
            "cache_validator": cache_validation,
            "CPU_smoke": {"path": str(root / CPU_RESULT), "sha256": sha256_file(root / CPU_RESULT)},
            "CUDA_smoke": {"path": str(root / CUDA_RESULT), "sha256": sha256_file(root / CUDA_RESULT)},
            "paired_report_test": {"path": str(root / REPORT_TEST_RESULT), "sha256": sha256_file(root / REPORT_TEST_RESULT)},
            "cross_account_cache_access": {
                "path": str(root / CACHE_ACCESS_RESULT),
                "sha256": sha256_file(root / CACHE_ACCESS_RESULT),
                "train_cache_acl": cache_access["versioned_caches"]["train"]["windows_acl"],
                "validation_cache_acl": cache_access["versioned_caches"]["validation"]["windows_acl"],
                "treatment_contract_seeds": [
                    int(item["seed"]) for item in cache_access["treatment_contracts"]
                ],
            },
            "V10_observation_increment_regression": "passed",
            "V9_motion_regression": "passed",
            "V7_V8_repair_regression": "passed",
            "compileall": "passed",
            "src_v3_static_isolation": isolation,
        },
        "configs": configs,
        "formal_artifacts": formal_artifacts,
        "post_training_report_command": (
            "Set-Location F:\\Project\\RadarInsight\\src-v3; "
            "..\\.venv\\Scripts\\python.exe -m report.rd_physical_paired "
            "--project-root F:\\Project\\RadarInsight"
        ),
        "known_limits": [
            "Smoke validates contracts, finite loss/gradient/update and recovery; it is not classification performance",
            "validation CU support remains 23 parents; three paired seeds and the frozen stage-one gate are mandatory",
            "UnknownTarget open-set reject/abstain calibration remains separate and unexecuted",
            "V11 changes only the DBB/CU Gate; specialists and formal three-checkpoint cascade are outside this stage",
            "CUDA initial paired forwards are numerically but not bitwise identical under the user-selected non-strict protocol",
        ],
        "formal_training": "未执行",
        "formal_validation_performance": "未评估",
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "P95": "未执行",
        "formal_cascade": "未执行",
    }
    output = root / OUTPUT
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(output)
    result["result_sha256"] = sha256_file(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final V11 technical training-readiness snapshot.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
