from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.core.data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_NORMALIZATION_SCHEMA_VERSION
from src.process.build_rd_physical_normalization_stats import build_rd_physical_normalization_stats


SCHEMA_VERSION = RD_FIVE_NORMALIZATION_SCHEMA_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_rd_physical_five_normalization_stats(
    project_root: Path,
    *,
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv",
    output_name: str = "expert1_rd_physical_train_db_normalization_five_v1.json",
    sample_stride: int = 257,
) -> dict[str, object]:
    processed_root = project_root / "data" / "processed"
    temporary_name = f".{output_name}.legacy-builder.tmp.json"
    temporary_path = processed_root / temporary_name
    legacy = build_rd_physical_normalization_stats(
        project_root,
        index_name=index_name,
        output_name=temporary_name,
        labels=RD_FIVE_LABELS,
        sample_stride=sample_stride,
    )
    payload = json.loads(temporary_path.read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION
    payload["scope"] = (
        "train-only parent/content-group balanced five-class physical RD normalization; "
        "pairing status is excluded from RD-only model inputs"
    )
    payload["labels"] = list(RD_FIVE_LABELS)
    payload["base_generator_code_sha256"] = payload.get("generator_code_sha256")
    payload["generator_code_sha256"] = _sha256_file(Path(__file__).resolve())
    payload["artifact_generator"] = {
        "wrapper": "src/process/build_rd_physical_five_normalization_stats.py",
        "base": "src/process/build_rd_physical_normalization_stats.py",
    }
    output_path = processed_root / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.unlink(missing_ok=True)
    return {**payload, "output_path": output_path.relative_to(project_root).as_posix(), "legacy_builder": legacy.get("output_path")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only five-class physical RD normalization statistics.")
    parser.add_argument("--index-name", default="expert1_rd_only_causal_window_index_v1_physical_v3_1.csv")
    parser.add_argument("--output-name", default="expert1_rd_physical_train_db_normalization_five_v1.json")
    parser.add_argument("--sample-stride", type=int, default=257)
    args = parser.parse_args()
    result = build_rd_physical_five_normalization_stats(
        Path.cwd(),
        index_name=args.index_name,
        output_name=args.output_name,
        sample_stride=args.sample_stride,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
