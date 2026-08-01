from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.configmanager import load_expert1_config
from .expert1_trainer import _build_expert1_loaders
from .pretrain_track_manifest import write_pretrain_parent_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the exact duplicate-safe TR-only train-parent manifest for track pretraining."
    )
    parser.add_argument("--expert1-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_expert1_config(args.expert1_config)
    loaders = _build_expert1_loaders(config, validate_csv_files=False)
    payload = write_pretrain_parent_manifest(
        args.output,
        loaders.train_loader.dataset.index_df,
        source_index=f"data/processed/{config.index_name}",
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).as_posix()),
                "parent_count": payload["parent_count"],
                "parent_ids_sha256": payload["parent_ids_sha256"],
                "source_index": payload["source_index"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
