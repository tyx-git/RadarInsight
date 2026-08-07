from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.process.build_rd_physical_feature_stats import build_rd_physical_feature_stats
from src.core.rd_physical_schema import PHYSICAL_RD_CLEAN_FEATURE_STATS_SCHEMA_VERSION
from src.core.data.rd_physical_window_dataset import RD_ONLY_QUALITY_NAMES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_rd_physical_clean_feature_stats(
    project_root: Path,
    *,
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv",
    normalization_stats_name: str = "expert1_rd_physical_train_db_normalization_v4.json",
    output_name: str = "expert1_rd_physical_feature_stats_v5_clean.json",
    batch_size: int = 8,
) -> dict[str, object]:
    """Create the clean seven-field quality artifact without pairing metadata.

    The physical feature values are identical to the audited v4 values; only
    the learnable quality schema/provenance changes.  The historical v4 file is
    never overwritten and remains available for explicit legacy checkpoint
    loading.
    """
    processed = project_root / "data" / "processed"
    temporary_name = f".{output_name}.legacy-intermediate.json"
    legacy = build_rd_physical_feature_stats(
        project_root,
        index_name=index_name,
        normalization_stats_name=normalization_stats_name,
        output_name=temporary_name,
        batch_size=batch_size,
    )
    temporary_path = processed / temporary_name
    payload = json.loads(temporary_path.read_text(encoding="utf-8"))
    payload["schema_version"] = PHYSICAL_RD_CLEAN_FEATURE_STATS_SCHEMA_VERSION
    payload["scope"] = "train-only clean RD physical statistics; pairing-derived canonical_time excluded"
    payload["quality_names"] = list(RD_ONLY_QUALITY_NAMES)
    payload["quality_metadata_names"] = list(RD_ONLY_QUALITY_NAMES)
    payload["quality_audit"] = {
        name: value for name, value in payload.get("quality_audit", {}).items()
        if name in RD_ONLY_QUALITY_NAMES
    }
    payload["generator_code_sha256"] = _sha256_file(Path(__file__).resolve())
    payload["test_evaluation_executed"] = False
    output_path = processed / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.unlink()
    return {**payload, "output_path": output_path.relative_to(project_root).as_posix(), "legacy_source": legacy.get("output_path")}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build train-only clean RD quality/statistics artifact.")
    parser.add_argument("--index-name", default="expert1_rd_only_causal_window_index_v1_physical_v3_1.csv")
    parser.add_argument("--normalization-stats-name", default="expert1_rd_physical_train_db_normalization_v4.json")
    parser.add_argument("--output-name", default="expert1_rd_physical_feature_stats_v5_clean.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_rd_physical_clean_feature_stats(Path.cwd(), index_name=args.index_name, normalization_stats_name=args.normalization_stats_name, output_name=args.output_name, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
