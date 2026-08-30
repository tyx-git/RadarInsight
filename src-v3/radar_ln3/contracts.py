from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import V3Config, resolve_project_path
from .data import RD_QUALITY_NAMES, RD_QUALITY_SCHEMA, require_causal_rd_quality_schema
from .runtime import sha256_file


def verify_formal_data_contract(project_root: Path, config: V3Config) -> dict[str, Any]:
    require_causal_rd_quality_schema(config.data.rd_quality_schema)
    if config.schema_version not in {"RADAR-LN3-V2", "RADAR-LN3-V3", "RADAR-LN3-V4"} or config.train.mode != "formal_supervised":
        raise ValueError("formal data contract verification requires RADAR-LN3-V2/V3/V4 formal_supervised config")
    index_path = resolve_project_path(project_root, config.data.index_path)
    contract_path = resolve_project_path(project_root, config.data.contract_path)
    stats_path = resolve_project_path(project_root, config.data.normalization_stats_path or "")
    for path in (index_path, contract_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(path)
    index_hash = sha256_file(index_path)
    contract_hash = sha256_file(contract_path)
    stats_hash = sha256_file(stats_path)
    expected = {
        "index": config.data.expected_index_sha256,
        "contract": config.data.expected_contract_sha256,
        "stats": config.data.expected_stats_sha256,
    }
    actual = {"index": index_hash, "contract": contract_hash, "stats": stats_hash}
    for name, value in actual.items():
        expected_value = expected[name]
        if expected_value is None or value.lower() != expected_value.lower():
            raise ValueError(f"formal {name} hash mismatch: expected={expected_value}, actual={value}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract_schema = config.data.contract_schema_version or config.schema_version
    if contract.get("schema_version") != expected_contract_schema:
        raise ValueError(f"formal config must consume the {expected_contract_schema} formal data contract")
    if contract.get("source_index_sha256", "").lower() != index_hash.lower():
        raise ValueError("formal data contract source index hash does not match the configured index")
    feature_contract = contract.get("feature_contract", {})
    if feature_contract.get("rd_quality_schema") != RD_QUALITY_SCHEMA:
        raise ValueError("formal data contract does not declare the repaired causal RD quality schema")
    if feature_contract.get("rd_quality_names") != list(RD_QUALITY_NAMES):
        raise ValueError("formal data contract RD quality names do not match the repaired model input")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("schema_version") != "RADAR-LN3-V2":
        raise ValueError("formal config must consume the verified RADAR-LN3-V2 train statistics")
    amplitude = dict(stats.get("amplitude", {}))
    center = float(amplitude.get("center"))
    scale = float(amplitude.get("scale"))
    if abs(center - config.data.rd_amplitude_center) > 1.0e-6 or abs(scale - config.data.rd_amplitude_scale) > 1.0e-6:
        raise ValueError("formal RD normalization values do not match the recorded train statistics")
    if stats.get("source_split") != "train" or stats.get("scope", "").find("no validation or test") < 0:
        raise ValueError("formal normalization statistics are not explicitly train-only")
    if config.schema_version in {"RADAR-LN3-V3", "RADAR-LN3-V4"}:
        if config.data.index_schema_version != "RADAR-LN3-V3":
            raise ValueError("V3 formal config must consume RADAR-LN3-V3 point rows")
        if contract.get("source_purity", {}).get("allowed_sources") != ["CQ08"]:
            raise ValueError("V3 formal contract is not CQ08-pure")
        if contract.get("time_contract", {}).get("dt_valid_field") != "tr_dt_valid":
            raise ValueError("V3 formal contract does not declare tr_dt_valid")
        if contract.get("training_contract", {}).get("loss_mode") != "causal_prefix":
            raise ValueError("V3 formal contract does not declare causal_prefix loss")
    if config.schema_version == "RADAR-LN3-V4" and config.data.contract_schema_version != "RADAR-LN3-V3":
        raise ValueError("V4 must explicitly reuse the V3 data contract")
    return {
        "index_path": str(index_path),
        "contract_path": str(contract_path),
        "stats_path": str(stats_path),
        "index_sha256": index_hash,
        "contract_sha256": contract_hash,
        "stats_sha256": stats_hash,
        "train_stats_parent_count": int(stats["parent_count"]),
        "rd_amplitude_center": center,
        "rd_amplitude_scale": scale,
    }


__all__ = ["sha256_file", "verify_formal_data_contract"]
