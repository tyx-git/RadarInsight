from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


V2_CONFIG = "config/RADAR-LN3/RADAR-LN3-V2-cq08-tr-rd-formal-supervised-config.json"
V3_CONFIG = "config/RADAR-LN3/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised-config.json"
V3_CONTRACT = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-data-contract.json"
V3_INDEX = "data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv"
V2_STATS = "data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    payload = json.loads((project_root / V2_CONFIG).read_text(encoding="utf-8"))
    payload["schema_version"] = "RADAR-LN3-V3"
    payload["config_name"] = "RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal-supervised"
    data = dict(payload["data"])
    data.update({
        "index_path": V3_INDEX,
        "contract_path": V3_CONTRACT,
        "index_schema_version": "RADAR-LN3-V3",
        "expected_index_sha256": sha256_file(project_root / V3_INDEX),
        "expected_contract_sha256": sha256_file(project_root / V3_CONTRACT),
        "expected_stats_sha256": sha256_file(project_root / V2_STATS),
        "dt_valid_field": "tr_dt_valid",
        "normalization_stats_path": V2_STATS,
    })
    payload["data"] = data
    model = dict(payload["model"])
    model["dt_valid_dim"] = 1
    payload["model"] = model
    train = dict(payload["train"])
    train.update({
        "checkpoint_path": "artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/best_model.pt",
        "last_checkpoint_path": "artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal/last_model.pt",
        "formal_artifact_dir": "artifacts/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-formal",
        "loss_mode": "causal_prefix",
        "checkpoint_schema": "RADAR-LN3-V3-formal-checkpoint-v1",
    })
    payload["train"] = train
    output = project_root / V3_CONFIG
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "generated",
        "config_path": V3_CONFIG,
        "config_sha256": sha256_file(output),
        "index_sha256": data["expected_index_sha256"],
        "contract_sha256": data["expected_contract_sha256"],
        "stats_sha256": data["expected_stats_sha256"],
        "loss_mode": train["loss_mode"],
        "checkpoint_schema": train["checkpoint_schema"],
        "artifact_dir": train["formal_artifact_dir"],
        "stats_reused": V2_STATS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the RADAR-LN3-V3 prefix/dt formal config.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
