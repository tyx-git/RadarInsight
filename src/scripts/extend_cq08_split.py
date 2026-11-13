"""向 CQ08 parent 级 split 清单追加经过明确批准的新 parent。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from radar_ln3.data.atomic import write_json  # noqa: E402


def _path(root: Path, value: str | Path) -> Path:
    target = (root / Path(value)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径越过项目根目录：{value}") from exc
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append explicitly assigned CQ08 parents to the product split"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("data/manifests/cq08_grouped_split.json"),
    )
    parser.add_argument(
        "--parent-id",
        action="append",
        required=True,
        help="可重复；格式为 cq08|track|<track_id>",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--reason",
        default="explicitly appended newly discovered parent; train-only extension",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    target = _path(root, args.split_path)
    if not target.is_file():
        raise FileNotFoundError(f"CQ08 split 文件不存在：{target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CQ08 split 根节点必须是对象")
    parent_splits = payload.get("parent_splits")
    if not isinstance(parent_splits, dict):
        raise ValueError("CQ08 split 缺少 parent_splits 对象")
    excluded = {
        str(value)
        for value in (payload.get("excluded_product_only_parent_ids") or [])
    }
    requested = sorted({str(value).strip() for value in args.parent_id if str(value).strip()})
    if not requested:
        raise ValueError("至少需要一个 parent-id")
    invalid = [value for value in requested if not value.startswith("cq08|track|")]
    if invalid:
        raise ValueError(f"parent-id 格式无效：{invalid}")
    if set(requested) & excluded:
        raise ValueError("不能同时把 excluded parent 加入正式 split")

    changed: list[str] = []
    unchanged: list[str] = []
    for parent_id in requested:
        old = parent_splits.get(parent_id)
        if old is None:
            parent_splits[parent_id] = str(args.split)
            changed.append(parent_id)
        elif str(old) != str(args.split):
            raise ValueError(
                f"parent 已有冲突 split：{parent_id}: {old} != {args.split}"
            )
        else:
            unchanged.append(parent_id)

    payload["parent_splits"] = {
        key: parent_splits[key] for key in sorted(parent_splits)
    }
    payload["parent_counts"] = {
        name: sum(1 for value in parent_splits.values() if value == name)
        for name in ("train", "validation", "test")
    }
    extensions = dict(payload.get("extensions") or {})
    assignments = list(extensions.get("parent_assignments") or [])
    by_parent = {
        str(row.get("parent_sample_id")): dict(row)
        for row in assignments
        if isinstance(row, dict) and row.get("parent_sample_id")
    }
    for parent_id in requested:
        by_parent[parent_id] = {
            "parent_sample_id": parent_id,
            "split_name": str(args.split),
            "reason": str(args.reason),
        }
    extensions["policy"] = "explicit_parent_assignment"
    extensions["parent_assignments"] = [
        by_parent[key] for key in sorted(by_parent)
    ]
    payload["extensions"] = extensions

    if target.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖 CQ08 split：{target}；请使用 --overwrite")
    write_json(target, payload, overwrite=True, label="split 文件")
    return {
        "path": str(target.relative_to(root).as_posix()),
        "split": str(args.split),
        "changed_parent_ids": changed,
        "unchanged_parent_ids": unchanged,
        "parent_counts": payload["parent_counts"],
        "parent_count": len(parent_splits),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"CQ08 split extension failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
