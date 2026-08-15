"""Prepare the V20 zero-delta liquid-dynamics paired configurations.

The builder copies the frozen V16 DBB-specialist input contract and changes
only the liquid invalid-dt mode plus the versioned paired/artifact metadata.
No index, normalization, raw payload, or RD cache is rebuilt here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.task_models import (
    LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    V20_DENSITY_PAIRED_PROTOCOL,
    V20_SKIP_PAIRED_PROTOCOL,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


SOURCE_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-"
    "dbb-specialist-config.json"
)
VARIANTS: dict[str, dict[str, str]] = {
    "skip": {
        "label": "V20-skip",
        "name_prefix": "RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip",
        "protocol": V20_SKIP_PAIRED_PROTOCOL,
        "treatment_mode": "skip",
    },
    "density": {
        "label": "V20r-density",
        "name_prefix": "RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted",
        "protocol": V20_DENSITY_PAIRED_PROTOCOL,
        "treatment_mode": "density_weighted",
    },
}


def _write_json_if_same(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V20 config: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _config_payload(
    source: dict[str, Any],
    *,
    variant: dict[str, str],
    arm: str,
    counterpart_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    if seed not in {42, 43, 44}:
        raise ValueError(f"V20 paired seed must be one of 42/43/44, got {seed}")
    name = f"{variant['name_prefix']}-{arm}-seed{seed}"
    payload = json.loads(json.dumps(source, ensure_ascii=False))
    payload["schema_version"] = LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
    payload["config_name"] = name
    payload["model"]["liquid_invalid_dt_update_mode"] = (
        "bounded_observation" if arm == "control" else variant["treatment_mode"]
    )
    artifact_slug = f"{variant['name_prefix']}-{arm}-seed{seed}-formal".replace(
        "config/RADAR-LN3/", ""
    )
    artifact = f"artifacts/{artifact_slug}"
    payload["train"].update(
        {
            "seed": seed,
            "checkpoint_schema": f"RADAR-LN3-V20-{variant['label'].lower()}-{arm}-checkpoint-v1",
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "paired_protocol": variant["protocol"],
            "paired_arm": arm,
            "paired_seed": seed,
            "paired_counterpart_config_path": (
                f"config/RADAR-LN3/{counterpart_name}-config.json"
            ),
            "dataloader_seed": seed,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
        }
    )
    payload["v20_experiment"] = {
        "label": variant["label"],
        "control_mode": "bounded_observation",
        "treatment_mode": variant["treatment_mode"],
        "data_contract_reused": "RADAR-LN3-V16 read-only",
        "seed": seed,
        "test": "未执行",
    }
    if variant["label"] == "V20r-density" and seed == 42:
        payload["v20_experiment"].update(
            {
                "shared_control_config_path": (
                    "config/RADAR-LN3/RADAR-LN3-V20-cq08-liquid-dbb-specialist-"
                    "zero-delta-skip-control-seed42-config.json"
                ),
                "shared_control_summary_path": (
                    "artifacts/RADAR-LN3-V20-cq08-liquid-dbb-specialist-"
                    "zero-delta-skip-control-seed42-formal/training-summary.json"
                ),
                "shared_control_retrained": False,
                "formal_training_scope": (
                    "density_weighted_treatment_only_against_completed_v20_shared_control"
                ),
            }
        )
    elif variant["label"] == "V20r-density":
        payload["v20_experiment"].update(
            {
                "multiseed_confirmation": True,
                "multiseed_seed_set": [42, 43, 44],
                "formal_training_scope": (
                    "paired_v16_core_control_and_density_treatment_multiseed_confirmation"
                ),
            }
        )
    return payload


def build(project_root: Path, variant_name: str = "skip") -> dict[str, Any]:
    root = project_root.resolve()
    if variant_name not in VARIANTS:
        raise ValueError(f"unsupported V20 variant: {variant_name}")
    variant = VARIANTS[variant_name]
    source_path = root / SOURCE_CONFIG
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_config = load_task_config(source_path)
    if (
        source_config.schema_version != "RADAR-LN3-V16"
        or source_config.task.name != "dbb_specialist"
        or source_config.model.liquid_invalid_dt_update_mode != "bounded_observation"
    ):
        raise ValueError("V20 requires the frozen V16 bounded-observation DBB specialist config")
    source_contract = verify_task_data_contract(root, source_config)
    configs: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for arm in ("control", "treatment"):
        counterpart_arm = "treatment" if arm == "control" else "control"
        counterpart_name = f"{variant['name_prefix']}-{counterpart_arm}-seed42"
        payload = _config_payload(
            source,
            variant=variant,
            arm=arm,
            counterpart_name=counterpart_name,
            seed=42,
        )
        path = root / "config/RADAR-LN3" / f"{payload['config_name']}-config.json"
        _write_json_if_same(path, payload)
        # Re-load through the formal parser so the generated pair cannot drift
        # from the runtime contract.
        loaded = load_task_config(path)
        if loaded.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION:
            raise ValueError(f"generated {arm} config has an unexpected schema")
        configs[arm] = path
        payloads[arm] = payload
    return {
        "status": "passed",
        "variant": variant["label"],
        "schema_version": LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
        "source_config": str(root / SOURCE_CONFIG),
        "source_config_sha256": sha256_file(root / SOURCE_CONFIG),
        "source_contract_hashes": source_contract["hashes"],
        "paired_configs": {
            arm: {
                "path": str(path),
                "sha256": sha256_file(path),
                "mode": payloads[arm]["model"]["liquid_invalid_dt_update_mode"],
                "protocol": payloads[arm]["train"]["paired_protocol"],
            }
            for arm, path in configs.items()
        },
        "data_contract_rebuilt": False,
        "normalization_rebuilt": False,
        "test_payload_access": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V20 liquid-dynamics paired configs.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="skip")
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
