"""Focused non-training smoke for the V20r seed43/44 confirmation pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from build.liquid_dynamics_v20_multiseed import (
    ARMS,
    NEW_TRAINING_SEEDS,
    READINESS_PATH,
    READINESS_SCHEMA,
    config_relative_path,
)
from radar_ln3.data import DT_REASON_ZERO_DELTA
from radar_ln3.task_models import (
    V20_DENSITY_PAIRED_PROTOCOL,
    build_paired_model_for_config,
    build_task_dataloader,
    build_task_dataset,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)
from train.tasks import verify_v20_training_readiness


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    readiness_path = root / READINESS_PATH
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("paired_protocol") != V20_DENSITY_PAIRED_PROTOCOL
        or readiness.get("new_training_seeds") != list(NEW_TRAINING_SEEDS)
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test_payload_access_count") != 0
        or readiness.get("test") != "未执行"
    ):
        raise ValueError("V20r multiseed readiness is invalid")

    seed_checks: dict[str, object] = {}
    control_logit_hashes: list[str] = []
    contract_hashes: dict[str, object] | None = None
    for seed in NEW_TRAINING_SEEDS:
        configs = {
            arm: load_task_config(root / config_relative_path(seed, arm))
            for arm in ARMS
        }
        contracts = {
            arm: verify_task_data_contract(root, configs[arm]) for arm in ARMS
        }
        if contracts["control"]["hashes"] != contracts["treatment"]["hashes"]:
            raise ValueError(f"seed{seed} pair data contracts differ")
        if contract_hashes is None:
            contract_hashes = contracts["control"]["hashes"]
        elif contract_hashes != contracts["control"]["hashes"]:
            raise ValueError("seed43/44 data contracts differ")

        readiness_bindings = {
            arm: verify_v20_training_readiness(
                root,
                root / config_relative_path(seed, arm),
                dict(contracts[arm]["hashes"]),
                paired_protocol=V20_DENSITY_PAIRED_PROTOCOL,
                paired_seed=seed,
            )
            for arm in ARMS
        }
        control, treatment, pairing = build_paired_model_for_config(
            root, configs["control"]
        )
        dataset = build_task_dataset(
            root, configs["control"], split_name=configs["control"].data.train_split_name
        )
        exposed_parent = next(
            parent_id
            for parent_id in dataset.parent_ids
            if dataset.base._groups[parent_id]["tr_dt_reason"]
            .astype(str)
            .eq("zero_delta_at_second_precision")
            .any()
        )
        parent_dataset = type(dataset)(
            dataset.base, dataset.spec, parent_ids=[exposed_parent]
        )
        batch = next(
            iter(build_task_dataloader(parent_dataset, batch_size=1, shuffle=False))
        )
        keys = (
            "tr_features",
            "tr_missing_mask",
            "tr_dt_seconds",
            "tr_dt_valid",
            "tr_dt_reason",
            "tr_motion_features",
            "tr_motion_valid_mask",
            "rd_frames",
            "rd_frame_mask",
            "rd_quality",
            "tr_padding_mask",
        )
        tensors = {key: batch[key] for key in keys if key in batch}
        control.eval()
        treatment.eval()
        with torch.inference_mode():
            control_logits = control(**tensors)["sequence_logits"]
            treatment_logits = treatment(**tensors)["sequence_logits"]
        active = ~tensors["tr_padding_mask"]
        exposed_seen = torch.cumsum(
            (tensors["tr_dt_reason"] == DT_REASON_ZERO_DELTA).to(torch.int64), dim=1
        ).gt(0)
        pre_exposure = active & (~exposed_seen)
        after_exposure = active & exposed_seen
        if not bool(pre_exposure.any()) or not bool(after_exposure.any()):
            raise RuntimeError(f"seed{seed} smoke parent lacks both exposure regions")
        control_hash = _tensor_sha256(control_logits)
        control_logit_hashes.append(control_hash)
        seed_checks[str(seed)] = {
            "config_seeds_match": all(
                configs[arm].train.seed == seed
                and configs[arm].train.paired_seed == seed
                and configs[arm].train.dataloader_seed == seed
                for arm in ARMS
            ),
            "readiness_hash_equal": (
                readiness_bindings["control"]["sha256"]
                == readiness_bindings["treatment"]["sha256"]
                == sha256_file(readiness_path)
            ),
            "shared_initial_state_exact": pairing[
                "shared_parameters_and_buffers_exactly_equal"
            ],
            "pre_exposure_logits_exact": bool(
                torch.equal(
                    control_logits[pre_exposure], treatment_logits[pre_exposure]
                )
            ),
            "post_exposure_logits_differ": bool(
                not torch.equal(
                    control_logits[after_exposure], treatment_logits[after_exposure]
                )
            ),
            "control_initial_logits_sha256": control_hash,
            "exposed_parent": exposed_parent,
        }

    checks = {
        "readiness_passed_and_test_sealed": True,
        "seed43_pair_passed": all(
            bool(value)
            for key, value in dict(seed_checks["43"]).items()
            if key not in {"control_initial_logits_sha256", "exposed_parent"}
        ),
        "seed44_pair_passed": all(
            bool(value)
            for key, value in dict(seed_checks["44"]).items()
            if key not in {"control_initial_logits_sha256", "exposed_parent"}
        ),
        "seed43_and_seed44_initial_logits_are_distinct": (
            len(set(control_logit_hashes)) == len(NEW_TRAINING_SEEDS)
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": "RADAR-LN3-V20-density-weighted-multiseed-smoke-v1",
        "checks": checks,
        "seed_checks": seed_checks,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_training_by_smoke": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("V20r multiseed contract smoke failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke V20r seed43/44 paired configs without training."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
