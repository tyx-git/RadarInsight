"""Fit and seal the exploratory L3 calibration artifact from validation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _policy(path: Path | None) -> Any:
    from radar_ln3.decision import L3CalibrationPolicy

    if path is None:
        return L3CalibrationPolicy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("policy"), dict):
        payload = payload["policy"]
    if not isinstance(payload, dict):
        raise ValueError("policy JSON must be an object")
    return L3CalibrationPolicy(**payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit an exploratory L3 calibration")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--l1-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l2-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l1-config-path", type=Path, default=None)
    parser.add_argument("--l2-config-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument(
        "--manifest-path", type=Path, default=Path("src/config/manifests.json")
    )
    parser.add_argument(
        "--feature-cache-manifest-path",
        type=Path,
        default=Path(
            "data/cq08/cache/features/l3/validation/feature-cache-manifest.json"
        ),
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/representations"),
        help="冻结 endpoint representation 缓存目录",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path("src/config/radar_ln3_policy.json"),
        help="已知类优先的 L3 operating policy",
    )
    parser.add_argument(
        "--decision-profile-json",
        type=Path,
        default=Path("src/config/radar_ln3_decision.json"),
        help="L3 决策 profile；只读取 src 内的正式配置",
    )
    parser.add_argument(
        "--minimum-audit-unknown-support",
        type=int,
        default=None,
        help="override the minimum independent audit Unknown parent support",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--l1-head-path",
        type=Path,
        default=None,
        help="independent L1 knownness head checkpoint",
    )
    parser.add_argument(
        "--l2-head-path",
        type=Path,
        default=None,
        help="independent L2 knownness head checkpoint",
    )
    parser.add_argument(
        "--l1-router-from-head",
        action="store_true",
        help="使用 L1 independent head 的 target/clutter score 作为路由概率",
    )
    parser.add_argument(
        "--l2-knownness-threshold-mode",
        choices=("shared", "per_class"),
        default="shared",
        help="L2 knownness 阈值使用 shared 或按类别独立拟合",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/radar_ln3_calibration.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.training import fit_l3_validation_calibration

    policy = _policy(args.policy_json)
    if args.minimum_audit_unknown_support is not None:
        if int(args.minimum_audit_unknown_support) < 1:
            raise ValueError("--minimum-audit-unknown-support must be positive")
        policy_payload = policy.to_dict()
        policy_payload["minimum_audit_unknown_support"] = int(
            args.minimum_audit_unknown_support
        )
        policy = type(policy)(**policy_payload)
    decision_profile_path = args.decision_profile_json
    if not decision_profile_path.is_absolute():
        decision_profile_path = args.project_root.resolve() / decision_profile_path
    decision_profile = json.loads(
        decision_profile_path.read_text(encoding="utf-8")
    )
    if not isinstance(decision_profile, dict):
        raise ValueError("decision profile JSON 必须是对象")
    if args.l1_head_path is not None and args.l2_head_path is not None:
        from radar_ln3.checkpoint import Checkpoint

        l1_head = Checkpoint.load(args.project_root.resolve(), args.l1_head_path)
        l2_head = Checkpoint.load(args.project_root.resolve(), args.l2_head_path)
        decision_profile["source_head_model_state_sha256"] = {
            "l1": str(l1_head.payload.get("model_state_sha256", "")),
            "l2": str(l2_head.payload.get("model_state_sha256", "")),
        }
    artifact = fit_l3_validation_calibration(
        args.project_root.resolve(),
        policy=policy,
        manifest_path=args.manifest_path,
        feature_cache_manifest_path=args.feature_cache_manifest_path,
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
        l1_router_from_head=bool(args.l1_router_from_head),
        l2_knownness_threshold_mode=str(args.l2_knownness_threshold_mode),
        decision_profile=decision_profile,
    )
    return {
        "path": str(args.output_path),
        "status": artifact["status"],
        "deployment_eligible": bool(artifact["deployment_eligible"]),
        "source_role": artifact["source_role"],
        "source_manifest_sha256": artifact["source_manifest_sha256"],
        "label_support": artifact["label_support"],
        "status_reasons": artifact["status_reasons"],
        "artifact_sha256": artifact["artifact_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"L3 calibration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
