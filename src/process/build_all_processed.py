from __future__ import annotations

import json
from pathlib import Path

from .build_training_index import build_training_index
from .raw_excel_manifest import write_manifests
from .standardize_rd import export_all as export_rd_all
from .standardize_tracks import export_all as export_track_all


def build_all(project_root: Path) -> dict:
    manifest_summary = write_manifests(project_root)
    track_summary = export_track_all(project_root)
    rd_summary = export_rd_all(project_root)
    trainable_summary = build_training_index(project_root)
    summary = {
        "manifest": manifest_summary,
        "tracks": track_summary,
        "raw_rd": rd_summary,
        "trainable": trainable_summary,
    }
    output_path = project_root / "data" / "processed" / "pipeline_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = build_all(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
