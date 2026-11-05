"""正式产品 cache、view 和 anchor 的小型契约测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from radar_ln3.cache import load_event_logit_anchor, save_event_logit_anchor
from radar_ln3.data.contracts import DatasetView


def _anchor_payload() -> dict[str, object]:
    return {
        "source_manifest_hash": "a" * 64,
        "sample_ids": ["parent:event-0", "parent:event-1"],
        "parent_sample_ids": ["parent", "parent"],
        "causal_event_indices": [0, 1],
        "split_names": ["train", "train"],
        "parent_ids": ["parent"],
        "parent_lengths": [2],
        "baseline_logits": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
        ),
    }


def test_event_anchor_aligns_by_event_identity(tmp_path: Path) -> None:
    anchor = save_event_logit_anchor(
        tmp_path,
        "data/cq08/cache/anchor.pt",
        _anchor_payload(),
    )
    loaded = load_event_logit_anchor(tmp_path, "data/cq08/cache/anchor.pt")
    values = loaded.values_for_events(
        ["parent:event-1", "parent:event-0"],
        ["parent", "parent"],
        [1, 0],
        ["train", "train"],
    )
    np.testing.assert_allclose(values, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="anchor 缺少事件"):
        loaded.values_for_events(
            ["parent:event-9"], ["parent"], [9], ["train"]
        )


def test_dataset_view_hash_binds_task_and_event_order() -> None:
    base = DatasetView(
        view_name="baseline_non_test_task",
        split_name="train",
        include_unknown=False,
        training_view=True,
        allowed_labels=("DroneTarget",),
        sample_ids=("parent:event-0", "parent:event-1"),
        parent_ids=("parent",),
        event_ids_by_parent={
            "parent": ("parent:event-0", "parent:event-1")
        },
        task_name="l2",
    )
    changed_task = DatasetView(
        **{**base.__dict__, "task_name": "l1"}
    )
    changed_order = DatasetView(
        **{
            **base.__dict__,
            "sample_ids": ("parent:event-1", "parent:event-0"),
        }
    )
    assert base.view_hash != changed_task.view_hash
    assert base.view_hash != changed_order.view_hash


_L3_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_L3_STATE_PATH = Path("artifacts/radar_ln3/pipeline/l3/state.json")
_L3_CACHE_OUTPUTS = {
    "01-head-fit-cache": "data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json",
    "02-validation-cache": "data/cq08/cache/features/l3/validation/feature-cache-manifest.json",
}


def _write_l3_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def l3_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    if shutil.which("pwsh") is None:
        pytest.skip("隔离 trainL3 回归需要 pwsh")
    root = tmp_path / "sandbox"
    scripts = root / "src/scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(_L3_SOURCE_ROOT / "scripts/trainL3.ps1", scripts / "trainL3.ps1")
    common = scripts / "common"
    common.mkdir()
    shutil.copy2(_L3_SOURCE_ROOT / "scripts/common/pipeline_inputs.ps1", common / "pipeline_inputs.ps1")
    (root / "src/config").mkdir(parents=True)
    shutil.copy2(_L3_SOURCE_ROOT / "config/radar_ln3_l3.json", root / "src/config/radar_ln3_l3.json")
    _write_l3_json(root / "src/config/manifests.json", {"synthetic": True})
    for output in _L3_CACHE_OUTPUTS.values():
        _write_l3_json(root / output, {"synthetic_cache": True})
    state = root / _L3_STATE_PATH
    _write_l3_json(state, {
        "schema_version": "RADAR-LN3-FORMAL-L3-PIPELINE-V4",
        "configuration": {
            "device": "cpu",
            "batch_size": 4,
            "head_batch_size": 32,
            "epochs": 30,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "package_root": "src",
            "profile_path": "src/config/radar_ln3_decision.json",
        },
        "updated_utc": "2025-01-01T00:00:00Z",
        "steps": {
            name: {
                "status": "completed", "outputs": [output], "exit_code": 0,
                "log_path": "", "command_path": "", "error": "",
            }
            for name, output in _L3_CACHE_OUTPUTS.items()
        },
    })
    historical_package = root / "tmp/L3"
    historical_package.mkdir(parents=True)
    (historical_package / "preserve.marker").write_text("historical", encoding="utf-8")
    return root, state


def _run_l3(root: Path, *arguments: str, stage: str = "cache") -> subprocess.CompletedProcess[str]:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("隔离 trainL3 回归需要 pwsh")
    env = {
        **os.environ,
        "PYTHONPATH": str(_L3_SOURCE_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "NO_COLOR": "1",
        **{key: str(root / "tmp") for key in ("TMP", "TEMP", "TMPDIR")},
    }
    return subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File",
         str(root / "src/scripts/trainL3.ps1"),
         "-Stage", stage, "-Device", "cpu", "-Resume",
         "-PythonPath", sys.executable, *arguments],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def test_train_l3_uses_formal_bundle_without_touching_historical_tmp(l3_sandbox) -> None:
    root, state = l3_sandbox
    before = json.loads(state.read_text(encoding="utf-8"))
    historical = root / "tmp/L3/preserve.marker"
    original_historical = historical.read_bytes()
    result = _run_l3(root)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "[trainL3] 续跑跳过：01-head-fit-cache" in output
    assert "[trainL3] 续跑跳过：02-validation-cache" in output
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["steps"] == before["steps"]
    assert historical.read_bytes() == original_historical
    assert {path.relative_to(root / "tmp/L3").as_posix() for path in (root / "tmp/L3").rglob("*")} == {"preserve.marker"}
    assert (root / "artifacts/radar_ln3/pipeline/l3").is_dir()


def test_train_l3_resume_rejects_training_parameter_change(l3_sandbox) -> None:
    root, state = l3_sandbox
    before = state.read_bytes()
    result = _run_l3(root, "-Epochs", "31")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "续跑参数不一致：epochs" in output
    assert state.read_bytes() == before


def test_train_l3_retrains_heads_when_endpoint_binding_changes(l3_sandbox) -> None:
    root, _ = l3_sandbox
    for relative in (
        "src/config/model.json",
        "src/config/model.json",
        "artifacts/radar_ln3/l1/l1.pt",
        "artifacts/radar_ln3/l2/l2.pt",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("initial", encoding="utf-8")
    (root / "src/scripts/train_l3_heads.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path); p.add_argument('--output-root',type=Path); p.add_argument('--representation-root',type=Path); a,_=p.parse_known_args()\n"
        "root=a.project_root\n"
        "for name in ('l1-knownness.pt','l2-knownness.pt','l1-knownness-summary.json','l2-knownness-summary.json'):\n"
        " q=root/a.output_root/name; q.parent.mkdir(parents=True,exist_ok=True); q.write_text('output',encoding='utf-8')\n"
        "for name in ('l1-train.pt','l1-validation.pt','l2-train.pt','l2-validation.pt'):\n"
        " q=root/a.representation_root/name; q.parent.mkdir(parents=True,exist_ok=True); q.write_text('representation',encoding='utf-8')\n",
        encoding="utf-8",
    )
    first = _run_l3(root, stage="heads")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "[trainL3] 开始：03-knownness-heads" in first.stdout
    second = _run_l3(root, stage="heads")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "[trainL3] 续跑跳过：03-knownness-heads" in second.stdout
    configuration_path = root / "src/config/radar_ln3_l3.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["knownness_training"]["l1"]["bce_unknown_weight"] = 1.4
    _write_l3_json(configuration_path, configuration)
    changed = _run_l3(root, stage="heads")
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert "[trainL3] 开始：03-knownness-heads" in changed.stdout
    saved = json.loads((root / _L3_STATE_PATH).read_text(encoding="utf-8"))
    assert saved["steps"]["03-knownness-heads"]["knownness_training"]["l1"]["bce_unknown_weight"] == 1.4
    command = json.loads((root / "artifacts/radar_ln3/pipeline/l3/commands/03-knownness-heads.json").read_text(encoding="utf-8"))
    assert "--l3-config-path" in command["arguments"]
    assert "--l1-bce-unknown-weight" not in command["arguments"]
    (root / "artifacts/radar_ln3/l1/l1.pt").write_text("changed endpoint", encoding="utf-8")
    third = _run_l3(root, stage="heads")
    assert third.returncode == 0, third.stdout + third.stderr
    assert "[trainL3] 开始：03-knownness-heads" in third.stdout


def test_train_l3_reset_waits_for_formal_pipeline_lock(l3_sandbox) -> None:
    root, state = l3_sandbox
    before = state.read_bytes()
    lock = root / "artifacts/radar_ln3/pipeline/l3/run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("pwsh"), "-NoProfile", "-NonInteractive", "-File",
         str(root / "src/scripts/trainL3.ps1"), "-Stage", "cache", "-Device", "cpu",
         "-ResetState", "-PythonPath", sys.executable],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode != 0
    assert state.read_bytes() == before
    assert lock.read_text(encoding="utf-8") == str(os.getpid())


def test_resumable_step_accepts_verified_reused_outputs(tmp_path: Path) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("续跑输出契约测试需要 pwsh")
    root = tmp_path / "pipeline-root"
    artifact = root / "data/cq08/cache/l2-baseline-anchor.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"existing verified anchor")
    helper = _L3_SOURCE_ROOT / "scripts/common/pipeline_inputs.ps1"
    helper_literal = str(helper).replace("'", "''")
    root_literal = str(root).replace("'", "''")
    command = (
        f". '{helper_literal}'; "
        f"$fresh = Test-PipelineOutputs -ProjectRoot '{root_literal}' "
        "-Outputs @('data/cq08/cache/l2-baseline-anchor.pt') "
        "-RunStartedUtc ([DateTime]::UtcNow.AddMinutes(1)); "
        "if ($fresh) { throw 'old output unexpectedly passed fresh check' }; "
        f"$reused = Test-PipelineOutputs -ProjectRoot '{root_literal}' "
        "-Outputs @('data/cq08/cache/l2-baseline-anchor.pt') "
        "-RunStartedUtc ([DateTime]::UtcNow.AddMinutes(1)) -AllowExisting; "
        "if (-not $reused) { throw 'verified existing output was rejected' }"
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    script = (_L3_SOURCE_ROOT / "scripts/train.ps1").read_text(encoding="utf-8")
    assert ') -Outputs @("data/cq08/cache/l2-baseline-anchor.pt") -Inputs @("src/config/manifests.json") -AllowExistingOutputs' in script
