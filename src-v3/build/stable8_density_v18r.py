"""Prepare the V18r stable8 plus density-weighted DBB paired configs.

V18r does not rebuild data.  It copies the frozen V18 stable8 Treatment
contract into both arms and changes only the zero-delta dynamics mode plus
versioned paired/artifact identity.  The Control config is a read-only
reference; formal training is permitted only for the Treatment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_VALUE_DIM,
    V18R_SOURCE_TREATMENT_CONFIG_PATH,
    V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
    V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


NAME_PREFIX = (
    "RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-"
    "micro-doppler8-stable-density"
)
CHECKPOINT_SCHEMA = "RADAR-LN3-V18r-dbb-stable8-density-paired-checkpoint-v1"


def config_relative_path(arm: str) -> Path:
    if arm not in {"control", "treatment"}:
        raise ValueError(f"unsupported V18r arm: {arm}")
    role = "control-reference" if arm == "control" else "treatment"
    return Path(
        f"config/RADAR-LN3/{NAME_PREFIX}-{role}-seed42-config.json"
    )


def artifact_relative_path(arm: str) -> Path:
    if arm not in {"control", "treatment"}:
        raise ValueError(f"unsupported V18r arm: {arm}")
    role = "control-reference" if arm == "control" else "treatment"
    return Path(f"artifacts/{NAME_PREFIX}-{role}-seed42-formal")


def _write_json_if_same(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V18r config: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _payload(source: dict[str, Any], arm: str) -> dict[str, Any]:
    counterpart = "treatment" if arm == "control" else "control"
    role = "control-reference" if arm == "control" else "treatment"
    counterpart_role = (
        "control-reference" if counterpart == "control" else "treatment"
    )
    name = f"{NAME_PREFIX}-{role}-seed42"
    artifact = artifact_relative_path(arm).as_posix()
    value = json.loads(json.dumps(source, ensure_ascii=False))
    value["schema_version"] = V18R_STABLE8_DENSITY_SCHEMA_VERSION
    value["config_name"] = name
    value["model"]["rd_physical_dim"] = 4
    value["model"]["micro_doppler_dim"] = BROADBAND_MICRO_DOPPLER_VALUE_DIM
    value["model"]["liquid_invalid_dt_update_mode"] = (
        "bounded_observation" if arm == "control" else "density_weighted"
    )
    value["train"].update(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "paired_protocol": V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
            "paired_arm": arm,
            "paired_seed": 42,
            "paired_counterpart_config_path": (
                f"config/RADAR-LN3/{NAME_PREFIX}-{counterpart_role}-"
                "seed42-config.json"
            ),
            "seed": 42,
            "dataloader_seed": 42,
            "allow_parameter_updates": arm == "treatment",
            "record_parent_order_hash": True,
            "save_every_epoch": True,
        }
    )
    value["v18r_experiment"] = {
        "scope": "DBB specialist development validation parent-final only",
        "source_feature_config": V18R_SOURCE_TREATMENT_CONFIG_PATH,
        "source_feature_arm": "V18 stable8 Treatment",
        "control_mode": "bounded_observation",
        "treatment_mode": "density_weighted",
        "both_arms_use_rd_physical4_and_micro_doppler_stable8": True,
        "control_is_read_only_reference": arm == "control",
        "data_rebuilt": False,
        "normalization_rebuilt": False,
        "test": "未执行",
    }
    return value


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_path = root / V18R_SOURCE_TREATMENT_CONFIG_PATH
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_config = load_task_config(source_path)
    if (
        source_config.schema_version != "RADAR-LN3-V18"
        or source_config.task.name != "dbb_specialist"
        or source_config.model.rd_physical_dim != 4
        or source_config.model.micro_doppler_dim
        != BROADBAND_MICRO_DOPPLER_VALUE_DIM
        or source_config.model.liquid_invalid_dt_update_mode
        != "bounded_observation"
        or source_config.train.paired_arm != "treatment"
    ):
        raise ValueError("V18r requires the frozen V18 stable8 Treatment config")
    source_contract = verify_task_data_contract(root, source_config)
    configs: dict[str, Any] = {}
    for arm in ("control", "treatment"):
        payload = _payload(source_payload, arm)
        path = root / config_relative_path(arm)
        _write_json_if_same(path, payload)
        loaded = load_task_config(path)
        if (
            loaded.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
            or loaded.train.paired_arm != arm
        ):
            raise ValueError(f"generated V18r {arm} config violates the parser contract")
        configs[arm] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "mode": loaded.model.liquid_invalid_dt_update_mode,
            "allow_parameter_updates": loaded.train.allow_parameter_updates,
        }
    return {
        "status": "passed",
        "schema_version": V18R_STABLE8_DENSITY_SCHEMA_VERSION,
        "paired_protocol": V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
        "source_v18_treatment_config": str(source_path.resolve()),
        "source_v18_treatment_config_sha256": sha256_file(source_path),
        "source_contract_hashes": source_contract["hashes"],
        "paired_configs": configs,
        "data_contract_rebuilt": False,
        "normalization_rebuilt": False,
        "test_payload_access_count": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V18r stable8 plus density-weighted paired configs."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
