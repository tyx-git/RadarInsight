from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check(src_v3_root: Path) -> dict[str, object]:
    violations: list[dict[str, str]] = []
    forbidden = (
        re.compile(r"\bfrom\s+src(?:\.|\s|$)"),
        re.compile(r"\bimport\s+src(?:\.|\s|$)"),
        re.compile(r"sys\s*\.\s*path"),
        re.compile(r"(?:importlib|SourceFileLoader|spec_from_file_location)"),
    )
    for path in sorted(src_v3_root.rglob("*.py")):
        if path.name == "static_isolation_check.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                violations.append({"file": str(path), "pattern": pattern.pattern})
    result = {"status": "passed" if not violations else "failed", "root": str(src_v3_root), "violations": violations}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that src-v3 has no runtime dependency on src.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    check(args.root.resolve())


if __name__ == "__main__":
    main()
