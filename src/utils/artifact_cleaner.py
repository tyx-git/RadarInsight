from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence


def clean_artifacts(root: str | Path, keep: Sequence[str]) -> list[str]:
    artifact_root = Path(root)
    keep_set = set(keep)
    removed: list[str] = []
    if not artifact_root.exists():
        return removed
    for child in artifact_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name in keep_set:
            continue
        if name.startswith('tmp_'):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(name)
            continue
        if name == 'calibrated':
            continue
        report = child / 'evaluation_report.json'
        if report.exists():
            try:
                import json
                metrics = json.loads(report.read_text(encoding='utf-8'))
                if float(metrics.get('accuracy', 0.0)) < 0.5:
                    shutil.rmtree(child, ignore_errors=True)
                    removed.append(name)
            except Exception:
                continue
    return removed
