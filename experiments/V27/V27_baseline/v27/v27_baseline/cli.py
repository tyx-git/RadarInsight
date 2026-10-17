"""Single V27 command-line entry point.

It intentionally offers explicit stages only.  ``train`` does not calibrate
or deploy, and no command implicitly reads test/Unknown payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


from radar_ln3.runtime import resolve_project_path, sha256_file

from .checkpoint import load_v27_checkpoint
from .data import V27TaskDataset
from .evaluation import evaluate_v27
from .l3 import (
    ACTION_SELECTION_METHOD,
    L3_SCHEMA_VERSION,
    make_unfitted_registry,
    verify_l3_registry,
)
from .protocol import V27_ALLOWED_TASKS, prepare_v27_manifest, v27_artifact_path
from .runtime import (
    build_v27_model,
    configured_device,
    load_v27_context,
    read_v27_sealed_json,
    write_v27_sealed_json,
)
from .training import train_v27_task


def _root(value: Path) -> Path:
    return value.resolve()


def _task(value: str) -> str:
    if value not in V27_ALLOWED_TASKS:
        raise ValueError(f"unsupported V27 task: {value}")
    return value


def _has_current_fitted_action_contract(registry: Mapping[str, Any]) -> bool:
    """Recognise the explicit action-calibration payload introduced by L3 v2.

    This is intentionally narrower than full registry validation: ``prepare``
    needs to distinguish a known, historical contract migration from a corrupt
    current artifact.  Full integrity remains the job of
    :func:`verify_l3_registry` below.
    """

    scores = registry.get("fitted_scores")
    selection = scores.get("action_selection") if isinstance(scores, Mapping) else None
    return (
        isinstance(selection, Mapping)
        and selection.get("method") == ACTION_SELECTION_METHOD
    )


def _refresh_unfitted_l3_registry(root: Path, *, manifest_sha256: str) -> dict[str, Any]:
    """Refresh only a stale unfitted registry after a manifest change.

    A fitted registry encodes a calibration result and therefore must never be
    rewritten implicitly.  An unfitted registry contains no fitted scores and
    is only a fail-closed placeholder, so rebinding it to the just-prepared
    system manifest is safe and removes an avoidable command-order trap.
    """

    context = load_v27_context(root)
    registry_path = resolve_project_path(
        root,
        context.config.l3.calibration_registry_path,
    )
    if not registry_path.exists():
        registry = make_unfitted_registry(system_manifest_sha256=manifest_sha256)
        write_v27_sealed_json(registry_path, registry)
        return {
            "action": "created_unfitted_registry",
            "registry_path": str(registry_path),
            "registry_sha256": registry["registry_sha256"],
        }

    registry = read_v27_sealed_json(registry_path)
    status = registry.get("status")
    schema_current = registry.get("schema_version") == L3_SCHEMA_VERSION
    if status == "fitted":
        recorded_manifest = str(registry.get("system_manifest_sha256", ""))
        action_contract_current = _has_current_fitted_action_contract(registry)
        # A fitted registry is terminal evidence for its original system
        # manifest.  Verify it against that manifest before deciding whether
        # it can remain active; never silently relabel old calibration as a
        # result for newly changed code.
        if schema_current and action_contract_current:
            verify_l3_registry(
                registry,
                expected_system_manifest_sha256=recorded_manifest,
            )
        if (
            recorded_manifest == manifest_sha256
            and schema_current
            and action_contract_current
        ):
            return {
                "action": "preserved_fitted_registry",
                "registry_path": str(registry_path),
                "registry_sha256": str(registry["registry_sha256"]),
            }
        archive_dir = root / "artifacts" / "archive" / "v27-l3-registries"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / (
            f"{registry_path.name}.stale-before-{sha256_file(registry_path)[:16]}"
        )
        if archive_path.exists():
            if sha256_file(archive_path) != sha256_file(registry_path):
                raise FileExistsError(
                    f"V27 L3 registry archive collision: {archive_path}"
                )
            registry_path.unlink()
        else:
            registry_path.replace(archive_path)
        replacement = make_unfitted_registry(system_manifest_sha256=manifest_sha256)
        write_v27_sealed_json(registry_path, replacement)
        return {
            "action": (
                "archived_incompatible_fitted_registry_and_created_unfitted_registry"
                if not schema_current or not action_contract_current
                else "archived_stale_fitted_registry_and_created_unfitted_registry"
            ),
            "registry_path": str(registry_path),
            "registry_sha256": replacement["registry_sha256"],
            "archived_registry_path": str(archive_path),
        }
    if status != "unfitted":
        raise ValueError("V27 L3 registry status is invalid")

    # Validate integrity and fixed schema before replacing the placeholder.
    # Its old manifest binding is deliberately not asserted: that is exactly
    # the stale state being repaired by this command.
    try:
        verify_l3_registry(
            registry,
            expected_system_manifest_sha256=str(
                registry.get("system_manifest_sha256", "")
            ),
        )
    except ValueError as exc:
        if "schema/manifest binding differs" not in str(exc):
            raise
        # An unfitted placeholder has no calibration evidence.  A schema
        # migration may safely replace it after its sealed payload hash was
        # verified by ``read_v27_sealed_json`` above.
        refreshed = make_unfitted_registry(system_manifest_sha256=manifest_sha256)
        write_v27_sealed_json(registry_path, refreshed)
        return {
            "action": "replaced_obsolete_unfitted_registry",
            "registry_path": str(registry_path),
            "registry_sha256": refreshed["registry_sha256"],
        }
    if registry.get("system_manifest_sha256") == manifest_sha256 and schema_current:
        return {
            "action": "kept_current_unfitted_registry",
            "registry_path": str(registry_path),
            "registry_sha256": str(registry["registry_sha256"]),
        }

    refreshed = make_unfitted_registry(system_manifest_sha256=manifest_sha256)
    write_v27_sealed_json(registry_path, refreshed)
    return {
        "action": "refreshed_stale_unfitted_registry",
        "registry_path": str(registry_path),
        "registry_sha256": refreshed["registry_sha256"],
    }


def _initialize_unfitted_l3_registry(root: Path) -> dict[str, Any]:
    """Create a fresh placeholder without ever replacing fitted calibration."""

    context = load_v27_context(root)
    path = resolve_project_path(root, context.config.l3.calibration_registry_path)
    if path.exists():
        existing = read_v27_sealed_json(path)
        status = existing.get("status")
        if status == "fitted":
            verify_l3_registry(
                existing,
                expected_system_manifest_sha256=context.manifest_sha256,
            )
            raise FileExistsError(
                "refusing to replace a fitted V27 L3 registry; create a new "
                "calibration artifact explicitly"
            )
        if status != "unfitted":
            raise ValueError("V27 L3 registry status is invalid")
        verify_l3_registry(
            existing,
            expected_system_manifest_sha256=str(
                existing.get("system_manifest_sha256", "")
            ),
        )

    registry = make_unfitted_registry(system_manifest_sha256=context.manifest_sha256)
    write_v27_sealed_json(path, registry)
    return {
        "status": "created_unfitted_l3_registry",
        "registry_path": str(path),
        "registry_sha256": registry["registry_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V27 complete L1-L2-L3 baseline entry point"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train")
    train.add_argument("--task", required=True, choices=sorted(V27_ALLOWED_TASKS))
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task", required=True, choices=sorted(V27_ALLOWED_TASKS))
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--seed", type=int)
    sub.add_parser("init-unfitted-l3")
    sub.add_parser("prepare-open-set")
    sub.add_parser("materialize-representations")
    train_heads = sub.add_parser("train-knownness")
    train_heads.add_argument("--task", required=True, choices=sorted(V27_ALLOWED_TASKS))
    train_heads.add_argument("--seed", type=int)
    train_quality = sub.add_parser("train-quality")
    train_quality.add_argument("--task", required=True, choices=sorted(V27_ALLOWED_TASKS))
    train_quality.add_argument("--seed", type=int)
    finalize_heads = sub.add_parser("finalize-heads")
    finalize_heads.add_argument("--task", required=True, choices=sorted(V27_ALLOWED_TASKS))
    finalize_heads.add_argument("--seed", type=int)
    materialize_scores = sub.add_parser("materialize-head-scores")
    materialize_scores.add_argument("--task", choices=sorted(V27_ALLOWED_TASKS))
    materialize_scores.add_argument("--role", choices=("representation_fit", "representation_development", "calibration"))
    materialize_scores.add_argument("--seed", type=int)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--seed", type=int)
    replay = sub.add_parser("evaluate-open-set")
    replay.add_argument("--seed", type=int)
    sub.add_parser("audit")
    args = parser.parse_args()
    root = _root(args.project_root)
    if args.command == "prepare":
        _, binding, path, digest = prepare_v27_manifest(root)
        registry = _refresh_unfitted_l3_registry(root, manifest_sha256=digest)
        result = {
            "status": "prepared",
            "manifest_path": str(path),
            "manifest_sha256": digest,
            "data_registry_sha256": binding["data_registry_sha256"],
            "l3_registry": registry,
        }
    elif args.command == "train":
        result = train_v27_task(
            load_v27_context(root),
            task=_task(args.task),
            seed=args.seed,
            epochs_override=args.epochs,
        )
    elif args.command == "evaluate":
        context = load_v27_context(root)
        task = _task(args.task)
        device = configured_device(context.config)
        model = build_v27_model(context, task=task, device=device)
        payload = load_v27_checkpoint(
            args.checkpoint,
            project_root=root,
            config=context.config,
            model=model,
            manifest=context.manifest,
            data_binding=context.data_binding,
            map_location=device,
            seed=args.seed,
        )
        dataset = V27TaskDataset(
            root,
            context.config,
            task=task,
            split_name=context.config.data.validation_split_name,
        )
        result = evaluate_v27(
            model,
            dataset,
            config=context.config,
            device=device,
            checkpoint_epoch=int(payload["epoch"]),
        )
        path = v27_artifact_path(
            root,
            context.config.artifacts.evaluation_report_template,
            int(payload["seed"]),
            task=task,
        )
        write_v27_sealed_json(
            path,
            {
                "status": "completed_known_class_evaluation_only",
                **result,
                "checkpoint_path": str(args.checkpoint),
            },
        )
        result = {"report_path": str(path), **result}
    elif args.command == "prepare-open-set":
        from .open_set_data import prepare_open_set_pools

        result = prepare_open_set_pools(load_v27_context(root))
    elif args.command == "materialize-representations":
        from .open_set import materialize_representation_caches

        result = materialize_representation_caches(load_v27_context(root))
    elif args.command in {"train-knownness", "train-quality"}:
        from .open_set import train_open_set_heads

        result = train_open_set_heads(
            load_v27_context(root),
            task=_task(args.task),
            component=("knownness" if args.command == "train-knownness" else "quality"),
            seed=args.seed,
        )
    elif args.command == "finalize-heads":
        from .open_set import finalize_open_set_heads

        result = finalize_open_set_heads(
            load_v27_context(root), task=_task(args.task), seed=args.seed
        )
    elif args.command == "materialize-head-scores":
        from .open_set import materialize_head_score_cache

        context = load_v27_context(root)
        tasks = (args.task,) if args.task else tuple(sorted(V27_ALLOWED_TASKS))
        roles = (args.role,) if args.role else (
            "representation_fit",
            "representation_development",
            "calibration",
        )
        result = {
            "status": "materialized",
            "scores": {
                f"{task}:{role}": _score_cache_summary(
                    materialize_head_score_cache(
                        context,
                        task=_task(task),
                        role=role,
                        seed=args.seed,
                    )
                )
                for task in tasks
                for role in roles
            },
        }
    elif args.command == "calibrate":
        from .open_set import calibrate_v27

        result = calibrate_v27(load_v27_context(root), seed=args.seed)
    elif args.command == "evaluate-open-set":
        from .open_set import evaluate_open_set_v27

        result = _open_set_replay_summary(
            evaluate_open_set_v27(load_v27_context(root), seed=args.seed)
        )
    elif args.command == "audit":
        from .open_set import audit_open_set_v27

        result = audit_open_set_v27(load_v27_context(root))
    else:
        result = _initialize_unfitted_l3_registry(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _score_cache_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the CLI result compact; tensors and parent lists stay on disk."""

    return {
        "task": str(payload["task"]),
        "role": str(payload["role"]),
        "seed": int(payload["seed"]),
        "parent_count": len(payload["parent_ids"]),
        "representation_cache_sha256": str(payload["representation_cache_sha256"]),
        "head_state_sha256": str(payload["head_state_sha256"]),
        "score_cache_sha256": str(payload["score_cache_sha256"]),
    }


def _open_set_replay_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep CLI replay output readable while records remain in the report."""

    return {
        key: payload[key]
        for key in (
            "status",
            "schema_version",
            "experiment_id",
            "role",
            "seed",
            "system_manifest_sha256",
            "l3_registry_sha256",
            "representation_cache_manifest_sha256",
            "calibration_pool_registry_sha256",
            "score_cache_sha256",
            "decision_summary",
            "safety",
            "unknown_classifier_training_exposure",
            "test_payload_access_count",
            "deployment_eligible",
            "report_path",
            "predictions_path",
        )
        if key in payload
    }


if __name__ == "__main__":
    main()