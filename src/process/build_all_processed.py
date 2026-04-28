from __future__ import annotations

import json
from pathlib import Path

from .build_training_index import build_training_index
from .build_expert1_index import build_expert1_index
from .build_system_index import build_system_index
from .raw_excel_manifest import write_manifests
from .standardize_cq08 import export_all as export_cq08_all
from .standardize_rd import export_all as export_rd_all
from .standardize_tracks import export_all as export_track_all


def build_all(project_root: Path) -> dict:
    manifest_summary = write_manifests(project_root)
    track_summary = export_track_all(project_root)
    rd_summary = export_rd_all(project_root)
    cq08_summary = export_cq08_all(project_root)
    trainable_summary = build_training_index(project_root)
    system_summary = build_system_index(project_root)
    expert1_summary = build_expert1_index(project_root)
    summary = {
        "manifest": manifest_summary,
        "tracks": track_summary,
        "raw_rd": rd_summary,
        "cq08": cq08_summary,
        "trainable": trainable_summary,
        "system": system_summary,
        "expert1": expert1_summary,
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
