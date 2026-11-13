"""Run the frozen L3 thresholds against the independent audit view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a frozen L3 calibration")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("artifacts/radar_ln3"))
    parser.add_argument("--audit-split-name", default="test")
    parser.add_argument(
        "--audit-evidence-path",
        type=Path,
        default=None,
        help="行级审计证据输出路径；默认 <root>/l3/audit-evidence.jsonl，report_l3_audit 可直接复用",
    )
    parser.add_argument(
        "--thresholds-frozen-before-access",
        action="store_true",
        help="required acknowledgement that audit data was not used for fitting",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="不把 audit 结果登记到正式 product manifest；非默认 Root 自动跳过",
    )
    parser.add_argument(
        "--formal-available",
        action="store_true",
        help="发布为正式可用状态；保留 audit 指标和原始 audit passed 字段",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="跳过数据文件 size/hash 校验（verify_files=False）；manifest 结构与事件路径边界校验仍会执行",
    )
    parser.add_argument("--skip-event-cache-validation", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.artifacts import is_default_artifact_root, l3_layout
    from radar_ln3.decision import load_l3_calibration
    from radar_ln3.model import load_model_config
    from radar_ln3.training import finalize_l3_audit

    root = args.project_root.resolve()
    layout = l3_layout(root, args.root)
    config = load_model_config(
        root, args.model_config_path, artifact_root=layout.root
    )
    register = not args.skip_register
    if register and not is_default_artifact_root(layout.root):
        print(
            json.dumps(
                {
                    "stage": "register_skipped",
                    "root": layout.root.as_posix(),
                    "reason": "非默认 Root 的 audit 结果不登记到正式 product manifest",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        register = False
    calibration = load_l3_calibration(root, layout.exploratory_calibration)
    return finalize_l3_audit(
        root,
        calibration,
        audit_manifest_path=layout.audit_manifest,
        audit_feature_cache_manifest_path=layout.audit_feature_cache,
        thresholds_frozen_before_access=bool(args.thresholds_frozen_before_access),
        audit_split_name=str(args.audit_split_name),
        product_manifest_path="src/config/manifests.json",
        device=str(args.device),
        batch_size=args.batch_size,
        output_path=layout.calibration,
        overwrite=bool(args.overwrite),
        formal_available=bool(args.formal_available),
        l1_head_path=layout.l1_head,
        l2_head_path=layout.l2_head,
        verify_files=not bool(args.skip_data_validation),
        validate_event_cache=not bool(args.skip_event_cache_validation),
        representation_root=layout.representations,
        model_config_path=args.model_config_path,
        l1_checkpoint_path=layout.l1_checkpoint,
        l2_checkpoint_path=layout.l2_checkpoint,
        audit_evidence_path=args.audit_evidence_path or layout.audit_evidence,
        config=config,
        register=register,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 audit failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
