from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from build.rd_physical_paired import _write_npz_new_or_identical
from radar_ln3.task_models import build_task_dataset, load_task_config, verify_task_data_contract


OUTPUT_NAME = "RADAR-LN3-V11-rd-physical-cache-access-test-result.json"
TRAIN_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-train-rd-physical-event-cache.npz")
VALIDATION_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-val-rd-physical-event-cache.npz")
EXPECTED_CACHE_HASHES = {
    "train": "7261431fe1e39465ad3eb11802f9972b09da62299c3465101dafcd03aac42ba5",
    "validation": "91776d1dfce1f4804d31e490a86339ada463647ffae3e65703e741f21817e7fc",
}
TREATMENT_CONFIGS = tuple(
    Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-treatment-seed{seed}-config.json"
    )
    for seed in (42, 43, 44)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows_acl(path: Path) -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    completed = subprocess.run(
        ["icacls", str(path)],
        check=True,
        capture_output=True,
    )
    output = completed.stdout
    inherited_rule_count = output.count(b"(I)")
    broad_read_allowed = b"BUILTIN\\Users:(I)(RX)" in output or b"Authenticated Users:(I)(M)" in output
    return {
        "protected": inherited_rule_count == 0,
        "inherited_rule_count": inherited_rule_count,
        "broad_read_allowed": broad_read_allowed,
    }


def _verify_actual_caches(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cache_results: dict[str, Any] = {}
    for split, relative_path in (("train", TRAIN_CACHE), ("validation", VALIDATION_CACHE)):
        path = project_root / relative_path
        with np.load(path, allow_pickle=False) as payload:
            sample_ids = np.asarray(payload["sample_ids"])
            values = np.asarray(payload["values"])
            valid = np.asarray(payload["valid"])
            fields = sorted(payload.files)
            event_count = int(sample_ids.shape[0])
        reuse_status = _write_npz_new_or_identical(path, sample_ids, values, valid)
        digest = _sha256(path)
        if digest != EXPECTED_CACHE_HASHES[split]:
            raise AssertionError(f"V11 {split} cache hash changed: {digest}")
        if reuse_status != "reused_identical":
            raise AssertionError(f"V11 {split} cache did not use its immutable reuse path: {reuse_status}")
        acl = _windows_acl(path)
        if acl is not None and (acl["protected"] or not acl["broad_read_allowed"]):
            raise AssertionError(f"V11 {split} cache is not broadly readable: {acl}")
        cache_results[split] = {
            "path": str(path),
            "sha256": digest,
            "npz_fields": fields,
            "event_count": event_count,
            "write_status": reuse_status,
            "windows_acl": acl if acl is not None else "not_applicable",
        }

    contract_results: list[dict[str, Any]] = []
    configs = [load_task_config(project_root / relative_path) for relative_path in TREATMENT_CONFIGS]
    for config, relative_path in zip(configs, TREATMENT_CONFIGS, strict=True):
        evidence = verify_task_data_contract(project_root, config)
        physical = dict(evidence.get("rd_physical") or {})
        if physical.get("hashes", {}).get("train_cache_sha256") != EXPECTED_CACHE_HASHES["train"]:
            raise AssertionError(f"Treatment contract did not verify the train cache: {relative_path}")
        contract_results.append(
            {
                "config": str(project_root / relative_path),
                "seed": config.train.seed,
                "data_contract_status": "passed",
            }
        )

    dataset = build_task_dataset(
        project_root,
        configs[0],
        split_name=configs[0].data.train_split_name,
        parent_limit=1,
    )
    item = dataset[0]
    dataset_result = {
        "parent_count": len(dataset),
        "sequence_length": int(item.tr_features.shape[0]),
        "rd_physical_shape": list(item.rd_physical_features.shape),
        "rd_physical_valid_shape": list(item.rd_physical_valid.shape),
        "finite": bool(np.isfinite(item.rd_physical_features.numpy()).all()),
    }
    if dataset_result["rd_physical_shape"][-1] != 4 or not dataset_result["finite"]:
        raise AssertionError(f"V11 bounded Dataset cache load failed: {dataset_result}")
    return cache_results, contract_results, dataset_result


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    smoke_dir = project_root / "src-v3" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    # Exercise the real cache destination because src-v3/smoke intentionally
    # has sandbox-specific ACLs that do not represent data/processed.
    probe = project_root / "data" / "processed" / f".radar-ln3-v11-cache-access-probe-{os.getpid()}.npz"
    sample_ids = np.asarray(["event-a", "event-b"], dtype=np.str_)
    values = np.asarray([[0.0, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 0.0]], dtype=np.float32)
    valid = np.asarray([True, False], dtype=np.bool_)
    try:
        first_status = _write_npz_new_or_identical(probe, sample_ids, values, valid)
        first_hash = _sha256(probe)
        second_status = _write_npz_new_or_identical(probe, sample_ids, values, valid)
        second_hash = _sha256(probe)
        with np.load(probe, allow_pickle=False) as payload:
            fields = sorted(payload.files)
            arrays_equal = bool(
                np.array_equal(payload["sample_ids"], sample_ids)
                and np.array_equal(payload["values"], values)
                and np.array_equal(payload["valid"], valid)
            )
        acl = _windows_acl(probe)
        if first_status != "created" or second_status != "reused_identical":
            raise AssertionError(f"unexpected write statuses: {first_status}, {second_status}")
        if first_hash != second_hash or not arrays_equal or fields != ["sample_ids", "valid", "values"]:
            raise AssertionError("direct cache publication changed bytes or arrays")
        if acl is not None and (acl["protected"] or not acl["broad_read_allowed"]):
            raise AssertionError(f"directly published Windows cache is not broadly readable: {acl}")
        cache_results, contract_results, dataset_result = _verify_actual_caches(project_root)
        result = {
            "status": "passed",
            "scope": "focused cache publication and Windows ACL inheritance; no formal training",
            "first_write_status": first_status,
            "second_write_status": second_status,
            "sha256_stable": True,
            "npz_fields": fields,
            "arrays_equal": arrays_equal,
            "windows_acl": acl if acl is not None else "not_applicable",
            "versioned_caches": cache_results,
            "treatment_contracts": contract_results,
            "bounded_dataset": dataset_result,
            "formal_training": "未执行",
            "test_evaluation": "未执行",
        }
    finally:
        probe.unlink(missing_ok=True)

    output = smoke_dir / OUTPUT_NAME
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify V11 cache publication and cross-account-readable ACL inheritance.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
