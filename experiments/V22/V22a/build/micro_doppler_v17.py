from __future__ import annotations

import json
from pathlib import Path
from typing import Any




















def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_json_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different versioned file: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return "created"































