from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from smoke.task_cu import run as run_specialist
from smoke.task_dbb_cu import run as run_gate


CONFIGS = (
    (7, "dbb_cu_gate", "dbb-cu-gate", run_gate),
    (7, "dbb_specialist", "dbb-specialist", run_specialist),
    (7, "cu_specialist", "cu-specialist", run_specialist),
    (8, "dbb_cu_gate", "dbb-cu-gate", run_gate),
    (8, "dbb_specialist", "dbb-specialist", run_specialist),
    (8, "cu_specialist", "cu-specialist", run_specialist),
)


def run(project_root: Path, *, parent_limit: int = 6) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("V7/V8 CUDA smoke requires an available CUDA device; CPU fallback is forbidden")
    device_name = torch.cuda.get_device_name(0)
    results: dict[str, object] = {}
    for version, task, stem, runner in CONFIGS:
        config = (
            project_root
            / "config"
            / "RADAR-LN3"
            / f"RADAR-LN3-V{version}-cq08-liquid-{stem}-config.json"
        )
        result = runner(project_root, config, parent_limit=parent_limit, output_suffix="-cuda")
        if result.get("status") != "passed" or result.get("device") != "cuda":
            raise AssertionError(f"V{version} {task} did not pass on CUDA: {result}")
        results[f"V{version}_{task}"] = result
        gc.collect()
        torch.cuda.empty_cache()
    payload: dict[str, object] = {
        "status": "passed",
        "device": "cuda",
        "device_name": device_name,
        "parent_limit": int(parent_limit),
        "task_results": results,
        "artifact_scope": "six bounded CUDA smoke runs under src-v3/smoke; no formal artifact",
        "formal_training": "未执行",
        "validation": "未执行",
        "test": "未执行",
    }
    path = project_root / "src-v3" / "smoke" / "RADAR-LN3-V7-V8-cuda-smoke-result.json"
    payload["result_path"] = str(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all six bounded V7/V8 task smokes on CUDA.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-limit", type=int, default=6)
    args = parser.parse_args()
    run(args.project_root.resolve(), parent_limit=max(2, int(args.parent_limit)))


if __name__ == "__main__":
    main()
