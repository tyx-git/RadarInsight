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
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json"),
    )
    parser.add_argument("--l1-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l2-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l1-config-path", type=Path, default=None)
    parser.add_argument("--l2-config-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument(
        "--audit-manifest-path",
        type=Path,
        required=True,
        help="manifest produced by freeze_l3_audit_manifest.py",
    )
    parser.add_argument(
        "--audit-feature-cache-manifest-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/representations"),
        help="冻结 endpoint representation 缓存目录",
    )
    parser.add_argument("--audit-split-name", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--l1-head-path", type=Path, default=None)
    parser.add_argument("--l2-head-path", type=Path, default=None)
    parser.add_argument(
        "--product-manifest-path",
        type=Path,
        default=Path("src/config/manifests.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json"),
    )
    parser.add_argument(
        "--thresholds-frozen-before-access",
        action="store_true",
        help="required acknowledgement that audit data was not used for fitting",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--formal-available",
        action="store_true",
        help="发布为正式可用状态；保留 audit 指标和原始 audit passed 字段",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.decision import load_l3_calibration
    from radar_ln3.training import finalize_l3_audit

    root = args.project_root.resolve()
    calibration = load_l3_calibration(root, args.calibration_path)
    result = finalize_l3_audit(
        root,
        calibration,
        audit_manifest_path=args.audit_manifest_path,
        audit_feature_cache_manifest_path=args.audit_feature_cache_manifest_path,
        thresholds_frozen_before_access=bool(args.thresholds_frozen_before_access),
        audit_split_name=str(args.audit_split_name),
        product_manifest_path=args.product_manifest_path,
        device=str(args.device),
        batch_size=args.batch_size,
        output_path=args.output_path,
        overwrite=bool(args.overwrite),
        l1_head_path=args.l1_head_path,
        l2_head_path=args.l2_head_path,
        verify_files=not bool(args.skip_data_validation),
        representation_root=args.representation_root,
        l1_checkpoint_path=args.l1_checkpoint_path,
        l2_checkpoint_path=args.l2_checkpoint_path,
        l1_config_path=args.l1_config_path,
        l2_config_path=args.l2_config_path,
        model_config_path=args.model_config_path,
    )
    if args.formal_available:
        from radar_ln3.checkpoint import canonical_hash

        output_path = args.output_path if args.output_path.is_absolute() else root / args.output_path
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["status"] = "formal_available"
        payload["deployment_eligible"] = True
        payload["research_only"] = False
        payload["formal_promotion"] = {
            "kind": "formal-audit-promotion",
            "source_status": result.get("status"),
            "audit_passed": bool((payload.get("audit") or {}).get("passed")),
        }
        payload["artifact_sha256"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "artifact_sha256"}
        )
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["status"] = "formal_available"
        result["deployment_eligible"] = True
    return result


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
