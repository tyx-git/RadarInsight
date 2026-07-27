from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.core.expert1_config import Expert1TrainConfig
from src.core.evaluate_expert1 import _grouped_bootstrap_summary
from src.core.expert1_data import ParentPairedBatchSampler, RDAwareBatchSampler
from src.core.expert1_trainer import _joint_logit_delta_probe
from src.core.expert1_trainer import _build_expert1_checkpoint_payload
from src.core.expert1_trainer import _checkpoint_joint_logit_delta_scale
from src.core.expert1_trainer import _locked_precision_parameters
from src.core.expert1_trainer import _load_checkpoint_into_modules
from src.core.expert1_trainer import _precision_logit_bias_from_checkpoint
from src.core.expert1_trainer import _expert1_release_gate
from src.core.expert1_trainer import _parent_window_aggregation_metrics
from src.core.expert1_trainer import _precision_candidate_delta_vs_selected
from src.core.expert1_trainer import _precision_candidate_from_diagnostic_checkpoint_tests
from src.core.expert1_trainer import _precision_candidate_readiness_summary
from src.core.expert1_trainer import _precision_calibration_probe
from src.core.expert1_trainer import _rd_joint_bottleneck_summary
from src.core.expert1_trainer import _select_final_test_checkpoint
from src.core.expert1_trainer import _expert1_selection_score
from src.core.expert1_trainer import _serializable_joint_logit_delta_probe
from src.core.expert1_trainer import _source_supported_metric_dict
from src.core.experts import LOW_ALTITUDE_CLASSES
from src.core.experts import LowAltitudeSuspectExpert
from src.core.metrics import compute_classification_metrics
from src.core.reporting import build_markdown_report
from src.core.reselect_expert1_precision import _apply_test_reporting
from src.core.reselect_expert1_precision import _precision_candidate_from_validation_records
from src.core.reselect_expert1_precision import _replace_precision_fields
from src.core.reselect_expert1_precision import _test_checkpoint_reporting
from src.core.reselect_expert1_precision import _write_precision_checkpoint
from src.core.system_trainer import _load_expert1_bundle


def _build_stage5_window_aggregation() -> dict[str, object]:
    unknown = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
    clutter = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
    sample_ids = [
        "parent_unknown|win0",
        "parent_unknown|win1",
        "parent_unknown|win2",
        "parent_clutter|win0",
        "parent_clutter|win1",
        "parent_clutter|win2",
    ]
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([unknown, unknown, unknown, clutter, clutter, clutter], dtype=torch.long)
    metadata: dict[str, dict[str, object]] = {}
    for index, sample_id in enumerate(sample_ids):
        parent_id = sample_id.split("|win", 1)[0]
        window_number = index % 3
        metadata[sample_id] = {
            "parent_sample_id": parent_id,
            "source_name": "raw_tr_rd",
            "label_name": LOW_ALTITUDE_CLASSES[int(labels[index].item())],
            "window_start_seconds": float(window_number),
            "window_end_seconds": float(window_number + 1),
        }
    return _parent_window_aggregation_metrics(
        {
            "sample_ids": sample_ids,
            "logits": logits,
            "labels": labels,
            "source_names": ["raw_tr_rd"] * len(sample_ids),
            "has_rd": [True] * len(sample_ids),
        },
        metadata,
    )


class Expert1PrecisionControlsSmokeTests(unittest.TestCase):
    def test_locked_precision_parameters_use_validation_only(self) -> None:
        validation_probe = {
            "ready": True,
            "selected_scale": 0.5,
            "recommended": {
                "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, -0.05],
            },
        }

        locked = _locked_precision_parameters(validation_probe, default_scale=1.0)

        self.assertEqual(locked["source"], "validation")
        self.assertAlmostEqual(float(locked["joint_logit_delta_scale"]), 0.5)
        self.assertEqual(locked["precision_logit_bias"], [0.0, 0.0, 0.0, 0.05, -0.05])

    def test_locked_precision_parameters_fall_back_without_ready_validation_probe(self) -> None:
        locked = _locked_precision_parameters({}, default_scale=0.75)

        self.assertEqual(locked["source"], "config_default")
        self.assertAlmostEqual(float(locked["joint_logit_delta_scale"]), 0.75)
        self.assertEqual(locked["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

    def test_validation_selected_scale_persists_with_config_default_zero_bias(self) -> None:
        locked = _locked_precision_parameters(
            {},
            default_scale=1.0,
            validation_selected_scale=0.5,
        )
        self.assertEqual(locked["source"], "validation_scale")
        self.assertEqual(locked["joint_logit_delta_scale"], 0.5)
        self.assertEqual(locked["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict={},
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="validation_selection",
            eligible=False,
            epoch=1,
            components={},
            joint_logit_delta_probe={"selected_scale": 0.5},
            locked_precision=locked,
            validation_precision_probe={"ready": False},
        )
        self.assertEqual(payload["selected_joint_logit_delta_scale"], 0.5)
        self.assertEqual(payload["config"]["joint_logit_delta_scale"], 0.5)
        self.assertEqual(payload["locked_precision_parameters"]["source"], "validation_scale")

    def test_legacy_validation_joint_probe_restores_selected_scale(self) -> None:
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict={},
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="legacy_validation_selection",
            eligible=False,
            epoch=1,
            components={},
            joint_logit_delta_probe={
                "enabled": True,
                "selected_scale": 0.5,
                "results": [{"scale": 0.5}],
            },
            locked_precision={
                "source": "config_default",
                "joint_logit_delta_scale": 1.0,
                "precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES),
            },
            validation_precision_probe={"ready": False},
        )

        self.assertEqual(payload["selected_joint_logit_delta_scale"], 1.0)
        self.assertEqual(
            _checkpoint_joint_logit_delta_scale(
                payload,
                Expert1TrainConfig(joint_logit_delta_scale=1.0),
            ),
            0.5,
        )

    def test_checkpoint_payload_preserves_candidate_role(self) -> None:
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict={},
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="joint_diagnostic",
            eligible=False,
            epoch=3,
            components={},
            joint_logit_delta_probe=None,
            locked_precision={
                "source": "config_default",
                "joint_logit_delta_scale": 1.0,
                "precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES),
            },
            validation_precision_probe=None,
            candidate_role="rd_diagnostic",
        )

        self.assertEqual(payload["candidate_role"], "rd_diagnostic")

    def test_validation_scale_provenance_rejects_nonzero_bias(self) -> None:
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict={},
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="validation_selection",
            eligible=False,
            epoch=1,
            components={},
            joint_logit_delta_probe=None,
            locked_precision={
                "source": "validation_scale",
                "joint_logit_delta_scale": 0.5,
                "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, 0.0],
            },
            validation_precision_probe=None,
        )
        self.assertEqual(payload["precision_calibration_source"], "config_default")
        self.assertEqual(payload["selected_joint_logit_delta_scale"], 1.0)
        self.assertEqual(payload["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

    def test_locked_precision_parameters_use_safe_defaults_for_invalid_validation_values(self) -> None:
        locked = _locked_precision_parameters(
            {
                "ready": True,
                "selected_scale": "invalid",
                "recommended": {"precision_logit_bias": [1.0]},
            },
            default_scale=0.75,
        )

        self.assertEqual(locked["source"], "config_default")
        self.assertAlmostEqual(float(locked["joint_logit_delta_scale"]), 0.75)
        self.assertEqual(locked["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

    def test_locked_precision_parameters_reject_non_finite_and_out_of_range_values(self) -> None:
        invalid_values = [float("nan"), float("inf"), float("-inf"), -0.01, 1.01]
        for invalid in invalid_values:
            with self.subTest(value=invalid):
                locked = _locked_precision_parameters(
                    {
                        "ready": True,
                        "selected_scale": invalid,
                        "recommended": {"precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES)},
                    },
                    default_scale=0.75,
                )
                self.assertEqual(locked["source"], "config_default")
                self.assertEqual(locked["joint_logit_delta_scale"], 0.75)
                self.assertEqual(locked["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

    def test_precision_bias_boundaries_reject_unauthorized_values(self) -> None:
        safe_bias = [0.0] * len(LOW_ALTITUDE_CLASSES)
        unauthorized_biases: list[object] = [
            [9.0] * len(LOW_ALTITUDE_CLASSES),
            [0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.10, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.20],
            [0.0, 0.0, 0.0, float("nan"), 0.0],
            [0.0, 0.0, 0.0, float("inf"), 0.0],
            [0.0] * (len(LOW_ALTITUDE_CLASSES) - 1),
            {"UnknownTarget": 0.05},
            object(),
        ]
        for invalid_bias in unauthorized_biases:
            with self.subTest(value=repr(invalid_bias)):
                locked = _locked_precision_parameters(
                    {
                        "ready": True,
                        "selected_scale": 0.5,
                        "recommended": {"precision_logit_bias": invalid_bias},
                    },
                    default_scale=0.75,
                )
                self.assertEqual(locked["source"], "config_default")
                self.assertEqual(locked["precision_logit_bias"], safe_bias)

                checkpoint_bias = _precision_logit_bias_from_checkpoint(
                    {
                        "precision_logit_bias": invalid_bias,
                        "config": {"precision_logit_bias": [0.0, 0.0, 0.0, 0.05, 0.15]},
                    }
                )
                self.assertEqual(checkpoint_bias, safe_bias)

    def test_checkpoint_bias_uses_only_authorized_nested_config_values(self) -> None:
        authorized_bias = [0.0, 0.0, 0.0, 0.05, 0.15]
        self.assertEqual(
            _precision_logit_bias_from_checkpoint(
                {"config": {"precision_logit_bias": authorized_bias}}
            ),
            authorized_bias,
        )

    def test_checkpoint_payload_rebuilds_caller_lock_and_nested_config(self) -> None:
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict={},
            config_payload={
                "joint_logit_delta_scale": 0.75,
                "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
            },
            source_feature_stats=None,
            selection_reason="validation_selection",
            eligible=True,
            epoch=1,
            components={},
            joint_logit_delta_probe=None,
            locked_precision={
                "source": "caller",
                "joint_logit_delta_scale": 0.5,
                "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
            },
            validation_precision_probe=None,
        )

        expected_bias = [0.0] * len(LOW_ALTITUDE_CLASSES)
        self.assertEqual(payload["precision_calibration_source"], "config_default")
        self.assertEqual(payload["precision_logit_bias"], expected_bias)
        self.assertEqual(payload["locked_precision_parameters"]["precision_logit_bias"], expected_bias)
        self.assertEqual(payload["config"]["precision_logit_bias"], expected_bias)
        self.assertEqual(payload["config"]["precision_calibration_source"], "config_default")

        for invalid in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(bias=invalid):
                locked = _locked_precision_parameters(
                    {
                        "ready": True,
                        "selected_scale": 0.5,
                        "recommended": {
                            "precision_logit_bias": [0.0, 0.0, 0.0, invalid, 0.0],
                        },
                    },
                    default_scale=0.75,
                )
                self.assertEqual(locked["source"], "config_default")
                self.assertEqual(locked["joint_logit_delta_scale"], 0.75)
                self.assertEqual(locked["precision_logit_bias"], [0.0] * len(LOW_ALTITUDE_CLASSES))

    def test_checkpoint_payload_behavior_persists_validation_locked_parameters(self) -> None:
        locked = {
            "source": "validation",
            "joint_logit_delta_scale": 0.5,
            "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, -0.05],
        }
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={"track": torch.tensor([1.0])},
            rd_proj_state_dict={"rd": torch.tensor([2.0])},
            model_state_dict={"model": torch.tensor([3.0])},
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="validation_selection",
            eligible=True,
            epoch=4,
            components={"macro_f1": 0.8},
            joint_logit_delta_probe={"selected_metrics": {"test_only": True}, "selected_scale": 0.5},
            locked_precision=locked,
            validation_precision_probe={"ready": True, "results": [{"private": True}]},
        )

        self.assertEqual(payload["selected_joint_logit_delta_scale"], locked["joint_logit_delta_scale"] )
        self.assertEqual(payload["precision_logit_bias"], locked["precision_logit_bias"] )
        self.assertEqual(payload["precision_calibration_source"], locked["source"] )
        self.assertEqual(payload["locked_precision_parameters"], locked)
        self.assertIs(payload["test_used_for_selection"], False)
        self.assertNotIn("selected_metrics", payload["joint_logit_delta_probe"] )
        self.assertNotIn("results", payload["validation_precision_calibration_probe"] )

    def test_checkpoint_round_trip_restores_validation_selected_scale_and_logits(self) -> None:
        from tempfile import TemporaryDirectory

        torch.manual_seed(31)
        original = LowAltitudeSuspectExpert(
            input_dim=4,
            rd_context_dim=0,
            use_rd_patch_cnn=True,
            traj_sequence_dim=16,
            hidden_dim=16,
            dropout=0.0,
            joint_logit_delta_scale=0.5,
        )
        original.eval()
        inputs = {
            "traj_input": torch.randn(2, 4),
            "rd_features": torch.randn(2, 2, 512, 4),
            "traj_sequence_features": torch.randn(2, 5, 16),
            "rd_patch_features": torch.randn(2, 2, 64, 64, 3),
            "rd_deep_patch_features": torch.randn(2, 2, 64, 64, 3),
            "rd_is_multimodal": torch.ones(2, dtype=torch.bool),
            "rd_alignment_quality": torch.tensor([0.15, 1.0]),
            "traj_padding_mask": torch.zeros(2, 5, dtype=torch.bool),
            "rd_padding_mask": torch.zeros(2, 2, dtype=torch.bool),
            "rd_patch_padding_mask": torch.zeros(2, 2, dtype=torch.bool),
            "rd_deep_patch_padding_mask": torch.zeros(2, 2, dtype=torch.bool),
            "enable_reject": False,
        }
        payload = _build_expert1_checkpoint_payload(
            representation_builder_state_dict={},
            rd_proj_state_dict={},
            model_state_dict=original.state_dict(),
            config_payload={"joint_logit_delta_scale": 1.0},
            source_feature_stats=None,
            selection_reason="validation_selection",
            eligible=False,
            epoch=1,
            components={},
            joint_logit_delta_probe={"selected_scale": 0.5},
            locked_precision={
                "source": "validation_scale",
                "joint_logit_delta_scale": 0.5,
                "precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES),
            },
            validation_precision_probe={"ready": False},
        )
        with torch.no_grad():
            expected_logits = original(**inputs).logits
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "round_trip.pt"
            torch.save(payload, checkpoint_path)
            restored = LowAltitudeSuspectExpert(
                input_dim=4,
                rd_context_dim=0,
                use_rd_patch_cnn=True,
                traj_sequence_dim=16,
                hidden_dim=16,
                dropout=0.0,
                joint_logit_delta_scale=1.0,
            )
            restored.eval()
            _load_checkpoint_into_modules(
                checkpoint_path,
                torch.nn.Identity(),
                torch.nn.Identity(),
                restored,
                torch.device("cpu"),
            )
            with torch.no_grad():
                restored_logits = restored(**inputs).logits
        self.assertEqual(restored.joint_logit_delta_scale, 0.5)
        self.assertTrue(torch.allclose(expected_logits, restored_logits, atol=1e-6, rtol=1e-6))

    def test_unrelated_test_probe_cannot_change_validation_locked_parameters(self) -> None:
        validation_probe = {
            "ready": True,
            "selected_scale": 0.5,
            "recommended": {
                "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, -0.05],
            },
        }
        fabricated_test_probe = {
            "ready": True,
            "selected_scale": 9.0,
            "recommended": {
                "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
            },
        }

        locked_before_test_reporting = _locked_precision_parameters(
            validation_probe,
            default_scale=1.0,
        )
        fabricated_test_probe["selected_scale"] = -7.0
        fabricated_test_probe["recommended"]["precision_logit_bias"] = [-7.0] * len(LOW_ALTITUDE_CLASSES)
        locked_after_test_reporting = _locked_precision_parameters(
            validation_probe,
            default_scale=1.0,
        )

        self.assertEqual(locked_after_test_reporting, locked_before_test_reporting)

    def test_test_reporting_path_cannot_export_or_select_calibration(self) -> None:
        root = Path(__file__).parents[1]
        trainer_path = root / "src/core/expert1_trainer.py"
        reselector_path = root / "src/core/reselect_expert1_precision.py"
        trainer_source = trainer_path.read_text(encoding="utf-8")
        reselector_source = reselector_path.read_text(encoding="utf-8")
        train_function = next(
            node for node in ast.parse(trainer_source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "train_expert1_v1"
        )
        actual_test_evaluations = [
            node for node in ast.walk(train_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_epoch"
            and any(
                isinstance(arg, ast.Attribute)
                and arg.attr == "test_loader"
                and isinstance(arg.value, ast.Name)
                and arg.value.id == "loaders"
                for arg in node.args
            )
        ]

        self.assertEqual(len(actual_test_evaluations), 1)
        protected_control_calls = {
            "_expert1_release_gate",
            "_rd_joint_bottleneck_summary",
        }
        for call in [node for node in ast.walk(train_function) if isinstance(node, ast.Call)]:
            name = call.func.id if isinstance(call.func, ast.Name) else ""
            if name in protected_control_calls:
                self.assertFalse(
                    any(isinstance(arg, ast.Name) and arg.id == "test_metrics" for arg in call.args),
                    f"{name} must consume validation control metrics, not test metrics",
                )
        self.assertIn('"report_scope": "test"', trainer_source)
        self.assertIn('"report_scope": "validation"', trainer_source)
        for source in (trainer_source, reselector_source):
            self.assertNotIn("precision_candidate_model_reselected.pt", source)
            self.assertNotIn("precision_candidate_model.pt", source)
        main_function = next(
            node for node in ast.parse(reselector_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calibration_calls = {
            "_precision_candidate_from_validation_records",
            "_joint_logit_delta_probe",
            "_precision_calibration_probe",
            "_write_precision_checkpoint",
        }
        for call in [node for node in ast.walk(main_function) if isinstance(node, ast.Call)]:
            name = call.func.id if isinstance(call.func, ast.Name) else ""
            if name in calibration_calls:
                self.assertFalse(
                    any(
                        isinstance(node, ast.Attribute) and node.attr == "test_loader"
                        for arg in call.args for node in ast.walk(arg)
                    ),
                    f"{name} must not consume test_loader",
                )

    def test_every_checkpoint_save_persists_validation_locked_calibration(self) -> None:
        trainer_source = Path(__file__).parents[1] / "src/core/expert1_trainer.py"
        source = trainer_source.read_text(encoding="utf-8")
        tree = ast.parse(source)
        train_function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "train_expert1_v1"
        )
        save_definition = next(
            node
            for node in train_function.body
            if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint"
        )
        parameter_names = [arg.arg for arg in save_definition.args.args]
        save_calls = [
            node
            for node in ast.walk(train_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_save_checkpoint"
        ]

        self.assertEqual(
            parameter_names[-2:],
            ["locked_precision", "validation_precision_probe"],
        )
        self.assertGreater(len(save_calls), 0)
        self.assertTrue(all(len(call.args) == len(parameter_names) for call in save_calls))
        self.assertIn('"precision_calibration_source": str(locked_precision["source"])', source)
        self.assertIn('"precision_logit_bias": precision_logit_bias', source)
        self.assertIn('"test_used_for_selection": False', source)

    def test_reselector_uses_validation_registry_and_exports_locked_metadata(self) -> None:
        with self.subTest("selection"):
            candidate = _precision_candidate_from_validation_records(
                {
                    "older": {
                        "path": "older.pt",
                        "epoch": 2,
                        "score": 0.7,
                        "selection_eligible": True,
                        "selected_joint_logit_delta_scale": 0.25,
                    },
                    "best_validation": {
                        "path": "best.pt",
                        "epoch": 3,
                        "score": 0.8,
                        "selection_eligible": True,
                        "locked_precision_parameters": {
                            "source": "validation",
                            "joint_logit_delta_scale": 0.5,
                            "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, -0.05],
                        },
                        "validation_precision_calibration_probe": {"ready": True, "probe_id": "best-validation"},
                    },
                }
            )
            self.assertEqual(candidate["checkpoint_name"], "best_validation")
            self.assertEqual(candidate["precision_calibration_source"], "validation")
            self.assertEqual(candidate["precision_calibration_probe"]["probe_id"], "best-validation")
            self.assertIs(candidate["test_used_for_selection"], False)

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "best.pt"
            destination = temp / "validation_precision_model.pt"
            torch.save(
                {
                    "selected_joint_logit_delta_scale": 1.0,
                    "config": {"joint_logit_delta_scale": 1.0},
                },
                source,
            )
            candidate["path"] = str(source)
            exported = _write_precision_checkpoint(candidate, destination)
            payload = torch.load(destination, map_location="cpu", weights_only=True)

            self.assertEqual(exported["status"], "exported")
            self.assertEqual(payload["selected_joint_logit_delta_scale"], 0.5)
            self.assertEqual(payload["precision_logit_bias"], candidate["precision_logit_bias"] )
            self.assertEqual(payload["precision_calibration_source"], "validation")
            self.assertEqual(payload["validation_precision_calibration_probe"]["probe_id"], "best-validation")
            self.assertIs(payload["test_used_for_selection"], False)
            self.assertEqual(payload["config"]["joint_logit_delta_scale"], 0.5)
            self.assertEqual(payload["config"]["precision_logit_bias"], candidate["precision_logit_bias"] )

    def test_reselector_excludes_rd_diagnostic_checkpoint(self) -> None:
        candidate = _precision_candidate_from_validation_records(
            {
                "release_candidate": {
                    "path": "release.pt",
                    "epoch": 3,
                    "score": 0.80,
                    "selection_eligible": False,
                    "candidate_role": "release",
                    "locked_precision_parameters": {
                        "source": "validation_scale",
                        "joint_logit_delta_scale": 0.5,
                        "precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES),
                    },
                },
                "best_joint_gain": {
                    "path": "diagnostic.pt",
                    "epoch": 4,
                    "score": 0.99,
                    "selection_eligible": True,
                    "candidate_role": "rd_diagnostic",
                    "locked_precision_parameters": {
                        "source": "validation_scale",
                        "joint_logit_delta_scale": 0.5,
                        "precision_logit_bias": [0.0] * len(LOW_ALTITUDE_CLASSES),
                    },
                },
            }
        )

        self.assertEqual(candidate["checkpoint_name"], "release_candidate")
        self.assertEqual(candidate["candidate_role"], "release")

    def test_reselector_export_rebuilds_all_locked_calibration_locations(self) -> None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.pt"
            destination = temp / "precision.pt"
            stale_probe = {"ready": True, "selected_scale": 0.9, "probe_id": "stale"}
            torch.save(
                {
                    "selected_joint_logit_delta_scale": 0.9,
                    "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
                    "precision_calibration_source": "validation",
                    "locked_precision_parameters": {
                        "source": "validation",
                        "joint_logit_delta_scale": 0.9,
                        "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
                    },
                    "validation_precision_calibration_probe": stale_probe,
                    "config": {
                        "joint_logit_delta_scale": 0.9,
                        "selected_joint_logit_delta_scale": 0.9,
                        "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
                    },
                },
                source,
            )
            bias = [0.0] * len(LOW_ALTITUDE_CLASSES)
            probe = {"ready": False, "probe_id": "registry-default"}
            exported = _write_precision_checkpoint(
                {
                    "path": str(source),
                    "checkpoint_name": "safe_default",
                    "recommended_joint_logit_delta_scale": 0.4,
                    "precision_logit_bias": bias,
                    "precision_calibration_source": "config_default",
                    "precision_calibration_probe": probe,
                },
                destination,
            )
            payload = torch.load(destination, map_location="cpu", weights_only=True)

        expected_locked = {
            "source": "config_default",
            "joint_logit_delta_scale": 1.0,
            "precision_logit_bias": bias,
        }
        self.assertEqual(payload["locked_precision_parameters"], expected_locked)
        self.assertEqual(payload["selected_joint_logit_delta_scale"], 1.0)
        self.assertEqual(payload["precision_logit_bias"], bias)
        self.assertEqual(payload["precision_calibration_source"], "config_default")
        self.assertNotIn("validation_precision_calibration_probe", payload)
        self.assertNotIn("precision_calibration_probe", payload)
        self.assertEqual(payload["config"]["joint_logit_delta_scale"], 1.0)
        self.assertEqual(payload["config"]["selected_joint_logit_delta_scale"], 1.0)
        self.assertEqual(payload["config"]["precision_logit_bias"], bias)
        self.assertEqual(exported["precision_calibration_source"], "config_default")

    def test_reselector_rejects_unsafe_registry_records_and_exports_safe_values(self) -> None:
        from tempfile import TemporaryDirectory
        invalid_records = {
            "nan_score": {"path": "nan.pt", "score": float("nan"), "epoch": 99, "selection_eligible": True},
            "inf_score": {"path": "inf.pt", "score": float("inf"), "epoch": 98, "selection_eligible": True},
            "bad_epoch": {"path": "epoch.pt", "score": 100.0, "epoch": "later", "selection_eligible": True},
            "bad_scale": {
                "path": "scale.pt", "score": 90.0, "epoch": 90, "selection_eligible": True,
                "locked_precision_parameters": {"source": "validation", "joint_logit_delta_scale": 1.5, "precision_logit_bias": [0.0] * 5},
            },
            "bad_bias": {
                "path": "bias.pt", "score": 80.0, "epoch": 80, "selection_eligible": True,
                "locked_precision_parameters": {"source": "validation", "joint_logit_delta_scale": 0.5, "precision_logit_bias": [0.0, 0.0, float("inf"), 0.0, 0.0]},
            },
            "safe": {
                "path": "safe.pt", "score": 0.8, "epoch": 4, "selection_eligible": True,
                "locked_precision_parameters": {"source": "validation", "joint_logit_delta_scale": 0.25, "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, -0.05]},
            },
        }
        candidate = _precision_candidate_from_validation_records(invalid_records)
        self.assertEqual(candidate["checkpoint_name"], "safe")
        self.assertTrue(torch.isfinite(torch.tensor(candidate["recommended_joint_logit_delta_scale"])).item())
        self.assertGreaterEqual(candidate["recommended_joint_logit_delta_scale"], 0.0)
        self.assertLessEqual(candidate["recommended_joint_logit_delta_scale"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(candidate["precision_logit_bias"])).all().item())

        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "safe.pt"
            destination = temp / "export.pt"
            torch.save({"config": {}}, source)
            candidate["path"] = str(source)
            _write_precision_checkpoint(candidate, destination)
            payload = torch.load(destination, map_location="cpu", weights_only=True)
        self.assertGreaterEqual(payload["selected_joint_logit_delta_scale"], 0.0)
        self.assertLessEqual(payload["selected_joint_logit_delta_scale"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(payload["precision_logit_bias"])).all().item())

    def test_post_lock_test_reporting_is_separate_and_cannot_mutate_validation_readiness(self) -> None:
        evaluation_report = {"checkpoint_selection": {}, "diagnostic_checkpoint_tests": {}}
        summary: dict[str, object] = {}
        validation_readiness = {
            "full_training_rerun_recommended_now": False,
            "next_action": "keep_validation_action",
        }
        _replace_precision_fields(
            evaluation_report,
            summary,
            {"checkpoint_name": "best", "path": "best.pt"},
            validation_readiness,
            {},
            {"status": "exported", "path": "precision.pt"},
            {"status": "metadata_only_not_evaluated"},
        )
        fabricated_test = {
            "status": "mismatch",
            "rd_joint_bottleneck_summary": {
                "full_training_rerun_recommended_now": True,
                "next_action": "test_must_not_override",
            },
        }
        reporting = _apply_test_reporting(evaluation_report, summary, fabricated_test)

        self.assertEqual(reporting["report_scope"], "test")
        self.assertIs(reporting["test_used_for_selection"], False)
        for holder in (evaluation_report, summary, evaluation_report["checkpoint_selection"]):
            self.assertEqual(holder["precision_candidate_readiness"], validation_readiness)
        self.assertEqual(evaluation_report["test_reporting"], reporting)
        self.assertEqual(summary["test_reporting"], reporting)
        self.assertNotIn("test_reporting", evaluation_report["checkpoint_selection"])
        self.assertEqual(
            _test_checkpoint_reporting({}, {"status": "exported"}),
            {"status": "no_precision_candidate"},
        )

    def test_parent_paired_batch_sampler_places_related_windows_together(self) -> None:
        parent_ids = ["a", "a", "b", "b", "c", "d", "e", "f"]
        sampler = ParentPairedBatchSampler(
            parent_ids=parent_ids,
            sample_weights=[1.0] * len(parent_ids),
            batch_size=4,
            pair_fraction=0.5,
            seed=9,
        )

        batches = list(iter(sampler))

        self.assertGreater(len(batches), 0)
        for batch in batches:
            counts: dict[str, int] = {}
            for index in batch:
                counts[parent_ids[index]] = counts.get(parent_ids[index], 0) + 1
            self.assertTrue(any(count >= 2 for count in counts.values()))

    def test_rd_aware_batch_sampler_keeps_rd_and_no_rd_in_stage2_batches(self) -> None:
        sampler = RDAwareBatchSampler(
            rd_indices=list(range(6)),
            no_rd_indices=list(range(100, 106)),
            sample_weights=[1.0] * 106,
            batch_size=4,
            rd_fraction=0.5,
            seed=3,
        )
        batches = list(iter(sampler))
        self.assertGreater(len(batches), 0)
        for batch in batches:
            rd_count = sum(index < 100 for index in batch)
            no_rd_count = sum(index >= 100 for index in batch)
            self.assertEqual(len(batch), 4)
            self.assertEqual(rd_count, 2)
            self.assertEqual(no_rd_count, 2)

    def test_rd_aware_batch_sampler_respects_pool_weights(self) -> None:
        sampler = RDAwareBatchSampler(
            rd_indices=[0, 1],
            no_rd_indices=[100, 101],
            sample_weights=[20.0, 1.0] + [1.0] * 98 + [25.0, 1.0],
            batch_size=2,
            rd_fraction=0.5,
            seed=9,
        )
        batches = list(iter(sampler))
        draws = [index for batch in batches for index in batch]
        self.assertGreater(draws.count(0), draws.count(1))
        self.assertGreater(draws.count(100), draws.count(101))

    def test_stage1_disabled_fusion_outputs_trajectory_final_even_with_rd(self) -> None:
        torch.manual_seed(11)
        model = LowAltitudeSuspectExpert(
            input_dim=4,
            rd_context_dim=0,
            use_rd_patch_cnn=False,
            traj_sequence_dim=16,
            hidden_dim=16,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                torch.randn(2, 4),
                rd_features=torch.randn(2, 3, 512, 4),
                traj_sequence_features=torch.randn(2, 5, 16),
                rd_is_multimodal=torch.tensor([True, True], dtype=torch.bool),
                traj_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
                rd_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
                enable_reject=False,
                disable_tr_rd_fusion=True,
            )
        self.assertFalse(bool(output.tr_rd_active.any().item()))
        self.assertTrue(torch.allclose(output.logits, output.traj_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output.joint_logits, output.traj_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(output.joint_rd_weight, torch.zeros_like(output.joint_rd_weight)))

    def test_supported_source_macro_ignores_labels_absent_from_source(self) -> None:
        logits = torch.tensor(
            [
                [5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        metrics = _source_supported_metric_dict(
            logits,
            labels,
            ["excel_track", "excel_track", "raw_tr_rd", "raw_tr_rd"],
        )
        self.assertEqual(metrics["excel_track"]["supported_class_count"], 2)
        self.assertAlmostEqual(metrics["excel_track"]["supported_macro_f1"], 1.0)
        self.assertEqual(metrics["raw_tr_rd"]["supported_class_count"], 2)
        self.assertGreater(metrics["raw_tr_rd"]["supported_macro_f1"], 0.0)

    def test_joint_logit_delta_probe_selects_middle_scale(self) -> None:
        traj_logits = torch.tensor(
            [
                [1.8, 2.0, 0.0, 0.0, 0.0],
                [2.0, 1.8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.4, 1.4],
                [0.0, 0.0, 0.0, 1.4, 2.4],
            ],
            dtype=torch.float32,
        )
        joint_logits = torch.tensor(
            [
                [2.4, 1.7, 0.0, 0.0, 0.0],
                [1.7, 2.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.8, 2.0],
                [0.0, 0.0, 0.0, 2.0, 1.8],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 3, 4], dtype=torch.long)
        traj_metrics = compute_classification_metrics(traj_logits, labels, num_classes=5)
        joint_metrics = compute_classification_metrics(joint_logits, labels, num_classes=5)
        base_metrics = {
            "traj_logits": traj_logits,
            "joint_logits": joint_logits,
            "rd_logits": joint_logits.clone(),
            "traj_cosine": traj_logits.clone(),
            "rd_cosine": joint_logits.clone(),
            "rd_pair_fusion_weights": torch.zeros((4, 3), dtype=torch.float32),
            "tr_rd_active": [True, True, True, True],
            "labels": labels,
            "source_names": ["cq08_track", "cq08_track", "excel_track", "excel_track"],
            "has_rd": [True, True, True, True],
            "traj_macro_f1": float(traj_metrics.macro_f1),
            "joint_macro_f1_on_rd": float(joint_metrics.macro_f1),
            "joint_minus_traj_macro_f1_on_rd": float(joint_metrics.macro_f1 - traj_metrics.macro_f1),
        }
        config = Expert1TrainConfig(
            joint_logit_delta_scale=1.0,
            joint_logit_delta_eval_scales=[0.0, 0.5, 1.0],
            joint_logit_delta_selection_metric="macro_f1_weak_class_guard",
            rd_proportional_fallback_quality=1.0,
        )

        probe = _joint_logit_delta_probe(base_metrics, config, stage2_enabled=True)

        self.assertTrue(probe["enabled"])
        self.assertAlmostEqual(float(probe["selected_scale"]), 0.5)
        self.assertEqual(len(probe["results"]), 3)
        by_scale = {float(item["scale"]): item for item in probe["results"]}
        self.assertLess(float(by_scale[0.0]["macro_f1"]), float(by_scale[0.5]["macro_f1"]))
        self.assertLess(float(by_scale[1.0]["macro_f1"]), float(by_scale[0.5]["macro_f1"]))
        serializable_probe = _serializable_joint_logit_delta_probe(probe)
        self.assertNotIn("selected_metrics", serializable_probe)

    def test_release_gate_rejects_test_rd_regression(self) -> None:
        config = Expert1TrainConfig()
        release_eligible, release_metrics, failures = _expert1_release_gate(
            {
                "macro_f1": 0.7750,
                "traj_macro_f1": 0.7886,
                "joint_minus_traj_macro_f1_on_rd": -0.0522,
                "rd_macro_f1_on_rd": 0.6096,
                "final_net_gain_rate_on_rd": -0.0527,
                "final_harm_rate_on_rd": 0.0699,
                "without_rd_accuracy": 0.9305,
                "parent_mean_logit_macro_f1": 0.8104,
                "parent_unknown_recall": 0.6667,
                "parent_clutter_recall": 0.5000,
                "parent_unknown_clutter_recall_mean": 0.5833,
                "final_per_class_recall": {
                    "UnknownTarget": 0.6548,
                    "ClutterTarget": 0.4500,
                },
            },
            config,
        )

        self.assertFalse(release_eligible)
        self.assertIn("joint_minus_traj_macro_f1_on_rd", failures)
        self.assertIn("final_net_gain_rate_on_rd", failures)
        self.assertIn("final_clutter_recall", failures)
        self.assertAlmostEqual(release_metrics["release_final_minus_traj_macro_f1"], -0.0136, places=4)

    def test_parent_window_aggregation_reports_stage5_diagnostics(self) -> None:
        aggregation = _build_stage5_window_aggregation()

        self.assertEqual(aggregation["parent_count"], 2)
        self.assertEqual(aggregation["window_count"], 6)
        flip = aggregation["prediction_flip_diagnostics"]
        self.assertEqual(flip["flipped_parent_count"], 2)
        self.assertAlmostEqual(float(flip["flip_count_mean"]), 1.0)
        self.assertAlmostEqual(float(flip["flip_rate_mean"]), 0.5)

        unknown = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
        clutter = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
        drone = LOW_ALTITUDE_CLASSES.index("DroneTarget")
        mean_confusion = aggregation["mean_logit"]["confusion_matrix"]
        self.assertEqual(mean_confusion[unknown][unknown], 1)
        self.assertEqual(mean_confusion[clutter][drone], 1)
        stable_confusion = aggregation["stable_decision"]["confusion_matrix"]
        self.assertEqual(stable_confusion[unknown][unknown], 1)
        self.assertEqual(stable_confusion[clutter][drone], 1)

        weak_recall = aggregation["weak_class_recall"]["mean_logit"]
        self.assertAlmostEqual(float(weak_recall["unknown_recall"]), 1.0)
        self.assertAlmostEqual(float(weak_recall["clutter_recall"]), 0.0)
        self.assertAlmostEqual(float(aggregation["parent_unknown_recall"]), 1.0)
        self.assertAlmostEqual(float(aggregation["parent_clutter_recall"]), 0.0)

        window_hard_pair = aggregation["unknown_clutter_hard_pair"]["window_to_parent_mean_logit"]
        self.assertEqual(window_hard_pair["support"], 6)
        self.assertEqual(window_hard_pair["rescue_count"], 1)
        self.assertEqual(window_hard_pair["harm_count"], 1)
        parent_rescue_harm = aggregation["parent_unknown_clutter_rescue_harm"]["first_window_to_mean_logit"]
        self.assertEqual(parent_rescue_harm["support"], 2)
        self.assertEqual(parent_rescue_harm["rescue_count"], 1)
        self.assertEqual(parent_rescue_harm["harm_count"], 1)

    def test_grouped_bootstrap_is_deterministic_and_uses_parent_groups(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        )
        metrics = {
            "logits": logits,
            "traj_logits": logits.clone(),
            "labels": torch.tensor([0, 1, 0, 3], dtype=torch.long),
            "sample_ids": [
                "cq08|track|1|win0000",
                "cq08|track|1|win0001",
                "excel|track|2|win0000",
                "excel|track|3|win0000",
            ],
            "source_names": ["cq08_track", "cq08_track", "excel_track", "excel_track"],
            "has_rd": [True, True, False, False],
        }

        first = _grouped_bootstrap_summary(metrics, replicates=25, seed=9)
        second = _grouped_bootstrap_summary(metrics, replicates=25, seed=9)

        self.assertEqual(first, second)
        self.assertTrue(bool(first["enabled"]))
        self.assertEqual(first["parent_group_count"], 3)
        self.assertEqual(first["source_parent_group_counts"], {"cq08_track": 1, "excel_track": 2})
        self.assertEqual(first["group_key"], "source_name|parent_sample_id")

    def test_final_test_checkpoint_prefers_highest_validation_score(self) -> None:
        selected = _select_final_test_checkpoint(
            best_path=Path("missing_best_model.pt"),
            best_epoch=6,
            best_score=3.99,
            best_reason="stage2_v2_joint_composite",
            best_components={"final_macro_f1": 0.75},
            best_selection_eligible=True,
            fallback_path=Path(__file__),
            fallback_epoch=17,
            fallback_score=4.06,
            fallback_components={"final_macro_f1": 0.75},
            checkpoint_registry={
                "fallback_candidate": {
                    "epoch": 17,
                    "score": 4.06,
                    "selection_reason": "stage2_v2_joint_composite",
                    "selection_eligible": False,
                }
            },
        )

        self.assertEqual(selected["name"], "fallback_candidate")
        self.assertEqual(selected["epoch"], 17)
        self.assertFalse(bool(selected["selection_eligible"]))

    def test_final_test_checkpoint_excludes_joint_diagnostic_candidate(self) -> None:
        selected = _select_final_test_checkpoint(
            config=Expert1TrainConfig(),
            best_path=Path(__file__),
            best_epoch=11,
            best_score=4.16,
            best_reason="stage2_v2_joint_composite",
            best_components={
                "final_macro_f1": 0.77,
                "traj_macro_f1": 0.75,
                "final_minus_traj_macro_f1": 0.02,
                "joint_minus_traj_macro_f1_on_rd": 0.01,
                "final_net_gain_rate_on_rd": 0.01,
                "final_harm_rate_on_rd": 0.04,
                "rd_macro_f1_on_rd": 0.56,
                "final_unknown_clutter_recall_mean": 0.52,
                "final_unknown_recall": 0.61,
                "final_clutter_recall": 0.43,
                "source_balanced_supported_macro_f1": 0.58,
            },
            best_selection_eligible=True,
            fallback_path=Path(__file__),
            fallback_epoch=11,
            fallback_score=4.16,
            fallback_components={},
            checkpoint_registry={
                "best_joint_gain": {
                    "epoch": 17,
                    "score": 0.032,
                    "selection_reason": "stage2_v2_joint_composite",
                    "selection_eligible": True,
                    "selection_components": {
                        "final_macro_f1": 0.79,
                        "traj_macro_f1": 0.73,
                        "final_minus_traj_macro_f1": 0.06,
                        "joint_minus_traj_macro_f1_on_rd": 0.03,
                        "final_net_gain_rate_on_rd": 0.05,
                        "final_harm_rate_on_rd": 0.01,
                        "rd_macro_f1_on_rd": 0.51,
                        "final_unknown_clutter_recall_mean": 0.58,
                        "final_unknown_recall": 0.72,
                        "final_clutter_recall": 0.43,
                        "source_balanced_supported_macro_f1": 0.62,
                    },
                }
            },
            candidate_paths={"best_joint_gain": Path(__file__)},
        )

        self.assertEqual(selected["name"], "best_selection")
        self.assertEqual(selected["epoch"], 11)
        self.assertEqual(selected["candidate_role"], "release")
        self.assertEqual(selected["selection_pool"], "release_eligible")

    def test_final_checkpoint_names_fallback_when_best_selection_was_never_saved(self) -> None:
        selected = _select_final_test_checkpoint(
            config=Expert1TrainConfig(),
            best_path=Path(__file__),
            best_epoch=4,
            best_score=0.71,
            best_reason="fallback_v2_selection_checkpoint",
            best_components={"final_macro_f1": 0.71},
            best_selection_eligible=False,
            fallback_path=Path(__file__),
            fallback_epoch=4,
            fallback_score=0.71,
            fallback_components={"final_macro_f1": 0.71},
            checkpoint_registry={
                "fallback_candidate": {
                    "epoch": 4,
                    "score": 0.71,
                    "selection_eligible": False,
                    "selection_components": {"final_macro_f1": 0.71},
                }
            },
        )

        self.assertEqual(selected["name"], "fallback_candidate")
        self.assertEqual(selected["candidate_role"], "fallback")
        self.assertEqual(selected["selection_pool"], "fallback_non_diagnostic")

    def test_final_test_checkpoint_uses_non_diagnostic_fallback_when_none_eligible(self) -> None:
        selected = _select_final_test_checkpoint(
            config=Expert1TrainConfig(),
            best_path=Path(__file__),
            best_epoch=9,
            best_score=0.72,
            best_reason="fallback_validation",
            best_components={
                "final_macro_f1": 0.72,
                "traj_macro_f1": 0.72,
                "final_net_gain_rate_on_rd": 0.0,
                "final_harm_rate_on_rd": 0.0,
            },
            best_selection_eligible=False,
            fallback_path=Path(__file__),
            fallback_epoch=10,
            fallback_score=0.74,
            fallback_components={
                "final_macro_f1": 0.74,
                "traj_macro_f1": 0.73,
                "final_net_gain_rate_on_rd": 0.01,
                "final_harm_rate_on_rd": 0.01,
            },
            checkpoint_registry={
                "fallback_candidate": {
                    "epoch": 10,
                    "score": 0.74,
                    "selection_eligible": False,
                    "selection_components": {
                        "final_macro_f1": 0.74,
                        "traj_macro_f1": 0.73,
                        "final_net_gain_rate_on_rd": 0.01,
                        "final_harm_rate_on_rd": 0.01,
                    },
                },
                "best_joint_gain": {
                    "epoch": 11,
                    "score": 0.99,
                    "selection_eligible": False,
                    "selection_components": {
                        "final_macro_f1": 0.82,
                        "traj_macro_f1": 0.70,
                        "joint_minus_traj_macro_f1_on_rd": 0.20,
                        "final_net_gain_rate_on_rd": 0.15,
                        "final_harm_rate_on_rd": 0.0,
                    },
                },
            },
            candidate_paths={"best_joint_gain": Path(__file__)},
        )

        self.assertEqual(selected["name"], "fallback_candidate")
        self.assertEqual(selected["candidate_role"], "fallback")
        self.assertEqual(selected["selection_pool"], "fallback_non_diagnostic")

    def test_selection_score_penalizes_rd_negative_gain(self) -> None:
        config = Expert1TrainConfig()
        baseline_score, baseline_eligible, baseline_components = _expert1_selection_score(
            {
                "macro_f1": 0.81,
                "accuracy": 0.88,
                "source_balanced_accuracy": 0.86,
                "min_source_accuracy": 0.80,
                "raw_tr_rd_accuracy": 0.90,
                "source_balanced_macro_f1": 0.80,
                "min_source_macro_f1": 0.78,
                "source_balanced_supported_macro_f1": 0.82,
                "min_source_supported_macro_f1": 0.79,
                "raw_tr_rd_macro_f1": 0.60,
                "raw_tr_rd_supported_macro_f1": 0.70,
                "excel_track_macro_f1": 0.62,
                "excel_track_supported_macro_f1": 0.68,
                "rd_raw_tr_rd_accuracy": 0.88,
                "rd_source_balanced_accuracy_on_rd": 0.82,
                "rd_min_source_accuracy_on_rd": 0.80,
                "rd_source_balanced_macro_f1_on_rd": 0.78,
                "rd_min_source_macro_f1_on_rd": 0.76,
                "without_rd_accuracy": 0.90,
                "traj_macro_f1": 0.79,
                "final_per_class_recall": {"UnknownTarget": 0.74, "ClutterTarget": 0.55},
                "final_unknown_recall": 0.74,
                "final_clutter_recall": 0.55,
                "parent_unknown_recall": 0.70,
                "parent_clutter_recall": 0.60,
                "parent_unknown_clutter_recall_mean": 0.65,
                "joint_macro_f1_on_rd": 0.80,
                "joint_minus_traj_macro_f1_on_rd": 0.02,
                "rd_macro_f1_on_rd": 0.62,
                "rd_drone_bird_recall_mean": 0.75,
                "rd_unknown_clutter_recall_mean": 0.64,
                "rd_balloon_recall": 0.70,
                "rd_clutter_recall": 0.58,
                "rd_unknown_recall": 0.66,
                "rd_harm_rate_on_rd": 0.02,
                "rd_net_gain_rate_on_rd": 0.01,
                "final_harm_rate_on_rd": 0.01,
                "final_net_gain_rate_on_rd": 0.02,
                "rd_residual_accuracy_on_rd": 0.80,
                "rd_unknown_clutter_detail_accuracy_on_residual_rd": 0.72,
                "rd_clutter_morphology_accuracy_on_residual_rd": 0.74,
                "residual_score_on_residual_mean": 0.60,
                "source_support": {"raw_tr_rd": 6, "excel_track": 10},
                "rd_source_support_on_rd": {"raw_tr_rd": 6},
                "rd_class_support_on_rd": {"UnknownTarget": 6, "ClutterTarget": 6, "BirdTarget": 6},
            },
            config,
            stage2_enabled=True,
        )
        penalized_score, penalized_eligible, penalized_components = _expert1_selection_score(
            {
                **{
                    "macro_f1": 0.81,
                    "accuracy": 0.88,
                    "source_balanced_accuracy": 0.86,
                    "min_source_accuracy": 0.80,
                    "raw_tr_rd_accuracy": 0.90,
                    "source_balanced_macro_f1": 0.80,
                    "min_source_macro_f1": 0.78,
                    "source_balanced_supported_macro_f1": 0.82,
                    "min_source_supported_macro_f1": 0.79,
                    "raw_tr_rd_macro_f1": 0.60,
                    "raw_tr_rd_supported_macro_f1": 0.70,
                    "excel_track_macro_f1": 0.62,
                    "excel_track_supported_macro_f1": 0.68,
                    "rd_raw_tr_rd_accuracy": 0.88,
                    "rd_source_balanced_accuracy_on_rd": 0.82,
                    "rd_min_source_accuracy_on_rd": 0.80,
                    "rd_source_balanced_macro_f1_on_rd": 0.78,
                    "rd_min_source_macro_f1_on_rd": 0.76,
                    "without_rd_accuracy": 0.90,
                    "traj_macro_f1": 0.79,
                    "final_per_class_recall": {"UnknownTarget": 0.74, "ClutterTarget": 0.55},
                    "final_unknown_recall": 0.74,
                    "final_clutter_recall": 0.55,
                    "parent_unknown_recall": 0.70,
                    "parent_clutter_recall": 0.60,
                    "parent_unknown_clutter_recall_mean": 0.65,
                    "joint_macro_f1_on_rd": 0.80,
                    "joint_minus_traj_macro_f1_on_rd": -0.02,
                    "rd_macro_f1_on_rd": 0.62,
                    "rd_drone_bird_recall_mean": 0.75,
                    "rd_unknown_clutter_recall_mean": 0.64,
                    "rd_balloon_recall": 0.70,
                    "rd_clutter_recall": 0.58,
                    "rd_unknown_recall": 0.66,
                    "rd_harm_rate_on_rd": 0.02,
                    "rd_net_gain_rate_on_rd": -0.01,
                    "final_harm_rate_on_rd": 0.01,
                    "final_net_gain_rate_on_rd": -0.02,
                    "rd_residual_accuracy_on_rd": 0.80,
                    "rd_unknown_clutter_detail_accuracy_on_residual_rd": 0.72,
                    "rd_clutter_morphology_accuracy_on_residual_rd": 0.74,
                    "residual_score_on_residual_mean": 0.60,
                    "source_support": {"raw_tr_rd": 6, "excel_track": 10},
                    "rd_source_support_on_rd": {"raw_tr_rd": 6},
                    "rd_class_support_on_rd": {"UnknownTarget": 6, "ClutterTarget": 6, "BirdTarget": 6},
                }
            },
            config,
            stage2_enabled=True,
        )

        self.assertGreater(baseline_score, penalized_score)
        self.assertEqual(baseline_components["selection_penalty_total"], 0.0)
        self.assertGreater(penalized_components["selection_penalty_total"], 0.0)
        self.assertLess(
            baseline_components["selection_penalty_total"],
            penalized_components["selection_penalty_total"],
        )
        self.assertIsInstance(baseline_eligible, bool)
        self.assertIsInstance(penalized_eligible, bool)

    def test_precision_candidate_promotes_high_accuracy_late_checkpoint(self) -> None:
        records = [
            {
                "checkpoint_name": "best_selection",
                "status": "ok",
                "epoch": 7,
                "is_selected_checkpoint": True,
                "macro_f1": 0.7834,
                "traj_macro_f1": 0.7896,
                "joint_minus_traj_macro_f1_on_rd": -0.0141,
                "rd_macro_f1_on_rd": 0.5359,
                "final_net_gain_rate_on_rd": -0.0103,
                "final_harm_rate_on_rd": 0.0218,
                "final_unknown_recall": 0.6190,
                "final_clutter_recall": 0.4500,
                "parent_mean_logit_macro_f1": 0.8077,
                "parent_unknown_recall": 0.7333,
                "parent_clutter_recall": 0.5000,
                "source_balanced_supported_macro_f1": 0.5900,
            },
            {
                "checkpoint_name": "last_full_val",
                "status": "ok",
                "epoch": 18,
                "is_selected_checkpoint": False,
                "macro_f1": 0.8151,
                "traj_macro_f1": 0.8020,
                "joint_minus_traj_macro_f1_on_rd": -0.0060,
                "rd_macro_f1_on_rd": 0.5900,
                "final_net_gain_rate_on_rd": 0.0137,
                "final_harm_rate_on_rd": 0.0046,
                "final_unknown_recall": 0.7857,
                "final_clutter_recall": 0.5500,
                "parent_mean_logit_macro_f1": 0.8077,
                "parent_unknown_recall": 0.7333,
                "parent_clutter_recall": 0.5000,
                "source_balanced_supported_macro_f1": 0.6052,
            },
        ]

        precision_candidate = _precision_candidate_from_diagnostic_checkpoint_tests(records)
        delta = _precision_candidate_delta_vs_selected(precision_candidate, records)

        self.assertEqual(precision_candidate["checkpoint_name"], "last_full_val")
        self.assertEqual(precision_candidate["precision_threshold_failures"], [])
        self.assertTrue(delta["selection_mismatch"])
        self.assertAlmostEqual(float(delta["macro_f1_delta"]), 0.0317, places=4)

    def test_precision_candidate_prefers_clutter_stability_when_macro_f1_is_close(self) -> None:
        records = [
            {
                "checkpoint_name": "unknown_heavy",
                "status": "ok",
                "epoch": 6,
                "is_selected_checkpoint": True,
                "macro_f1": 0.8060,
                "traj_macro_f1": 0.8060,
                "joint_minus_traj_macro_f1_on_rd": -0.0050,
                "rd_macro_f1_on_rd": 0.6250,
                "final_net_gain_rate_on_rd": 0.0,
                "final_harm_rate_on_rd": 0.0,
                "final_unknown_recall": 0.8200,
                "final_clutter_recall": 0.4600,
                "parent_mean_logit_macro_f1": 0.8200,
                "parent_unknown_recall": 0.8000,
                "parent_clutter_recall": 0.5000,
                "source_balanced_supported_macro_f1": 0.6100,
            },
            {
                "checkpoint_name": "clutter_stable",
                "status": "ok",
                "epoch": 7,
                "is_selected_checkpoint": False,
                "macro_f1": 0.8000,
                "traj_macro_f1": 0.8000,
                "joint_minus_traj_macro_f1_on_rd": -0.0050,
                "rd_macro_f1_on_rd": 0.6200,
                "final_net_gain_rate_on_rd": 0.0,
                "final_harm_rate_on_rd": 0.0,
                "final_unknown_recall": 0.7200,
                "final_clutter_recall": 0.5200,
                "parent_mean_logit_macro_f1": 0.8200,
                "parent_unknown_recall": 0.8000,
                "parent_clutter_recall": 0.5000,
                "source_balanced_supported_macro_f1": 0.6100,
            },
        ]

        precision_candidate = _precision_candidate_from_diagnostic_checkpoint_tests(records)

        self.assertEqual(precision_candidate["checkpoint_name"], "clutter_stable")
        self.assertEqual(precision_candidate["precision_threshold_failures"], [])
        self.assertAlmostEqual(float(precision_candidate["precision_candidate_clutter_recall_gap"]), -0.0200, places=4)
        self.assertAlmostEqual(float(precision_candidate["precision_candidate_unknown_recall_gap"]), -0.0200, places=4)
        self.assertAlmostEqual(float(precision_candidate["precision_candidate_joint_gain_gap"]), -0.0450, places=4)

    def test_precision_readiness_marks_small_unknown_gap_before_rerun(self) -> None:
        records = [
            {
                "checkpoint_name": "best_selection",
                "status": "ok",
                "epoch": 6,
                "is_selected_checkpoint": False,
                "selected_joint_logit_delta_scale": 1.0,
                "macro_f1": 0.7868,
                "traj_macro_f1": 0.8006,
                "joint_minus_traj_macro_f1_on_rd": -0.0342,
                "rd_macro_f1_on_rd": 0.6251,
                "final_net_gain_rate_on_rd": -0.0252,
                "final_harm_rate_on_rd": 0.0412,
                "final_unknown_recall": 0.8690,
                "final_clutter_recall": 0.4000,
                "parent_mean_logit_macro_f1": 0.8225,
                "source_balanced_supported_macro_f1": 0.5999,
                "conservative_precision_scale": 0.0,
                "conservative_precision_macro_f1": 0.8006,
                "conservative_precision_traj_macro_f1": 0.8006,
                "conservative_precision_joint_minus_traj_macro_f1_on_rd": -0.0342,
                "conservative_precision_final_net_gain_rate_on_rd": 0.0,
                "conservative_precision_final_harm_rate_on_rd": 0.0,
                "conservative_precision_final_unknown_recall": 0.8214,
                "conservative_precision_final_clutter_recall": 0.4667,
                "conservative_precision_source_balanced_supported_macro_f1": 0.5999,
            },
            {
                "checkpoint_name": "last_full_val",
                "status": "ok",
                "epoch": 18,
                "is_selected_checkpoint": False,
                "selected_joint_logit_delta_scale": 1.0,
                "macro_f1": 0.7744,
                "traj_macro_f1": 0.8048,
                "joint_minus_traj_macro_f1_on_rd": -0.0249,
                "rd_macro_f1_on_rd": 0.5354,
                "final_net_gain_rate_on_rd": -0.0069,
                "final_harm_rate_on_rd": 0.0355,
                "final_unknown_recall": 0.7024,
                "final_clutter_recall": 0.4500,
                "parent_mean_logit_macro_f1": 0.8242,
                "source_balanced_supported_macro_f1": 0.5874,
                "conservative_precision_scale": 0.25,
                "conservative_precision_macro_f1": 0.8034,
                "conservative_precision_traj_macro_f1": 0.8048,
                "conservative_precision_joint_minus_traj_macro_f1_on_rd": -0.0249,
                "conservative_precision_final_net_gain_rate_on_rd": 0.0034,
                "conservative_precision_final_harm_rate_on_rd": 0.0046,
                "conservative_precision_final_unknown_recall": 0.6905,
                "conservative_precision_final_clutter_recall": 0.5500,
                "conservative_precision_source_balanced_supported_macro_f1": 0.5962,
            },
        ]

        precision_candidate = _precision_candidate_from_diagnostic_checkpoint_tests(records)
        readiness = _precision_candidate_readiness_summary(records, precision_candidate)

        self.assertEqual(precision_candidate["checkpoint_name"], "last_full_val")
        self.assertFalse(bool(readiness["existing_scale_ready"]))
        self.assertFalse(bool(readiness["full_training_rerun_recommended_now"]))
        self.assertEqual(
            readiness["next_action"],
            "run_stage6c_unknown_recall_threshold_diagnostic_before_full_training",
        )
        self.assertEqual(readiness["closest_checkpoint_name"], "last_full_val")
        self.assertIn("precision_candidate_unknown_recall", readiness["closest_threshold_failures"])
        self.assertAlmostEqual(float(readiness["closest_threshold_gaps"]["precision_candidate_unknown_recall_gap"]), 0.0095, places=4)

    def test_precision_calibration_probe_can_recover_small_unknown_gap(self) -> None:
        unknown = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
        clutter = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
        labels = torch.tensor([0, 1, 2, unknown, unknown, clutter, clutter], dtype=torch.long)
        traj_logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.96, 1.00],
                [0.0, 0.0, 0.0, 1.00, 0.96],
                [0.0, 0.0, 0.0, 1.20, 0.10],
                [0.0, 0.0, 0.0, 1.10, 0.10],
            ],
            dtype=torch.float32,
        )
        base_metrics = {
            "traj_logits": traj_logits,
            "joint_logits": traj_logits,
            "rd_logits": traj_logits,
            "traj_cosine": torch.ones_like(traj_logits),
            "rd_cosine": torch.ones_like(traj_logits),
            "rd_pair_fusion_weights": torch.zeros((labels.shape[0], 3), dtype=torch.float32),
            "tr_rd_active": [False] * int(labels.shape[0]),
            "labels": labels,
            "source_names": ["excel_track"] * int(labels.shape[0]),
            "has_rd": [False] * int(labels.shape[0]),
            "traj_macro_f1": 0.7600,
            "joint_minus_traj_macro_f1_on_rd": 0.0,
            "rd_macro_f1_on_rd": 0.0,
        }
        precision_candidate = {
            "checkpoint_name": "last_full_val",
            "recommended_joint_logit_delta_scale": 0.0,
            "parent_mean_logit_macro_f1": 0.8500,
        }

        probe = _precision_calibration_probe(
            base_metrics,
            precision_candidate,
            Expert1TrainConfig(joint_logit_delta_eval_scales=[0.0]),
            stage2_enabled=False,
            window_metadata={},
        )

        self.assertTrue(bool(probe["ready"]))
        recommended = probe["recommended"]
        self.assertGreaterEqual(float(recommended["final_unknown_recall"]), 0.7000)
        self.assertGreaterEqual(float(recommended["final_clutter_recall"]), 0.5000)
        self.assertGreaterEqual(float(recommended["macro_f1"]), 0.8000)
        self.assertTrue(
            bool({"UnknownTarget", "ClutterTarget"} & set(recommended["precision_logit_bias_by_label"].keys()))
        )

    def test_rd_joint_bottleneck_summary_identifies_negative_rd_fusion(self) -> None:
        unknown = LOW_ALTITUDE_CLASSES.index("UnknownTarget")
        clutter = LOW_ALTITUDE_CLASSES.index("ClutterTarget")
        drone = LOW_ALTITUDE_CLASSES.index("DroneTarget")
        bird = LOW_ALTITUDE_CLASSES.index("BirdTarget")
        labels = torch.tensor([unknown, unknown, clutter, clutter, drone, bird], dtype=torch.long)

        def logits_for(preds: list[int]) -> torch.Tensor:
            logits = torch.full((len(preds), len(LOW_ALTITUDE_CLASSES)), -4.0)
            for row, pred in enumerate(preds):
                logits[row, pred] = 4.0
            return logits

        traj_logits = logits_for([unknown, unknown, clutter, clutter, drone, bird])
        rd_logits = logits_for([clutter, clutter, unknown, unknown, bird, bird])
        joint_logits = logits_for([clutter, unknown, unknown, clutter, drone, bird])
        final_logits = logits_for([unknown, unknown, clutter, clutter, drone, bird])
        config = Expert1TrainConfig(
            min_release_joint_minus_traj_macro_f1_on_rd=0.015,
            min_release_rd_evidence_macro_f1_on_rd=0.62,
            min_release_final_net_gain_rate_on_rd=0.015,
        )
        final_metrics = compute_classification_metrics(final_logits, labels, len(LOW_ALTITUDE_CLASSES))
        traj_metrics = compute_classification_metrics(traj_logits, labels, len(LOW_ALTITUDE_CLASSES))
        joint_metrics = compute_classification_metrics(joint_logits, labels, len(LOW_ALTITUDE_CLASSES))
        rd_metrics = compute_classification_metrics(rd_logits, labels, len(LOW_ALTITUDE_CLASSES))
        metrics = {
            "logits": final_logits,
            "traj_logits": traj_logits,
            "joint_logits": joint_logits,
            "rd_logits": rd_logits,
            "labels": labels,
            "source_names": ["cq08_track"] * 4 + ["raw_tr_rd"] * 2,
            "has_rd": [True] * 6,
            "rd_time_sync_status": ["proportional_fallback"] * 4 + ["exact_time_sync"] * 2,
            "macro_f1": final_metrics.macro_f1,
            "traj_macro_f1": traj_metrics.macro_f1,
            "traj_macro_f1_on_rd": traj_metrics.macro_f1,
            "joint_macro_f1_on_rd": joint_metrics.macro_f1,
            "joint_minus_traj_macro_f1_on_rd": joint_metrics.macro_f1 - traj_metrics.macro_f1,
            "rd_macro_f1_on_rd": rd_metrics.macro_f1,
            "final_net_gain_rate_on_rd": 0.0,
            "final_harm_rate_on_rd": 0.0,
            "selected_joint_logit_delta_scale": 0.25,
        }

        summary = _rd_joint_bottleneck_summary(metrics, config)

        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["primary_bottleneck"], "rd_evidence_below_release_and_joint_fusion_negative")
        self.assertFalse(summary["full_training_rerun_recommended_now"])
        self.assertIn("unknown_clutter_rd_worse_than_traj", summary["diagnosis_flags"])
        self.assertIn("proportional_fallback_joint_negative", summary["diagnosis_flags"])
        self.assertGreater(summary["release_gaps"]["rd_macro_f1_on_rd_gap"], 0.0)
        self.assertEqual(summary["hard_pair_bottlenecks"][0]["pair_name"], "UnknownTarget__vs__ClutterTarget")

    def test_rd_joint_bottleneck_summary_recommends_training_after_stage7_shift(self) -> None:
        metrics = {
            "has_rd": [True] * 873,
            "rd_hard_pair_report": {
                "UnknownTarget__vs__ClutterTarget": {
                    "support": 125,
                    "traj_accuracy": 0.6640,
                    "rd_accuracy": 0.1520,
                    "final_accuracy": 0.6400,
                    "rd_rescue_count": 5,
                    "rd_harm_count": 69,
                    "rd_net_gain_count": -64,
                    "rd_net_gain_rate": -0.5120,
                    "final_rescue_count": 1,
                    "final_harm_count": 4,
                    "final_net_gain_count": -3,
                    "final_net_gain_rate": -0.0240,
                }
            },
            "rd_time_sync_metrics": {
                "proportional_fallback": {
                    "support": 869,
                    "macro_f1": 0.7590,
                    "traj_macro_f1": 0.7587,
                    "joint_macro_f1": 0.7446,
                    "joint_minus_traj_macro_f1": -0.0141,
                    "rd_macro_f1": 0.4239,
                }
            },
            "macro_f1": 0.8021,
            "traj_macro_f1": 0.8048,
            "traj_macro_f1_on_rd": 0.7558,
            "joint_macro_f1_on_rd": 0.7416,
            "joint_minus_traj_macro_f1_on_rd": -0.0143,
            "rd_macro_f1_on_rd": 0.4227,
            "final_net_gain_rate_on_rd": 0.0034,
            "final_harm_rate_on_rd": 0.0069,
            "final_unknown_recall": 0.6786,
            "final_clutter_recall": 0.5333,
            "selected_joint_logit_delta_scale": 0.25,
        }
        config = Expert1TrainConfig(
            min_release_joint_minus_traj_macro_f1_on_rd=0.015,
            min_release_rd_evidence_macro_f1_on_rd=0.62,
            min_release_final_net_gain_rate_on_rd=0.015,
        )

        summary = _rd_joint_bottleneck_summary(metrics, config)

        self.assertTrue(summary["full_training_rerun_recommended_now"])
        self.assertEqual(summary["next_action"], "run_full_training_on_stage7_v3_rd_segment_store")
        self.assertIn(
            "stage7_rd_preprocess_distribution_shift_requires_training",
            summary["diagnosis_flags"],
        )

    def test_posthoc_test_observations_do_not_override_validation_readiness(self) -> None:
        evaluation_report = {
            "checkpoint_selection": {},
            "diagnostic_checkpoint_tests": {"results": []},
        }
        summary: dict[str, object] = {}
        precision_readiness = {
            "full_training_rerun_recommended_now": False,
            "next_action": "run_stage6c_unknown_recall_threshold_diagnostic_before_full_training",
        }
        test_observations = {
            "rd_joint_bottleneck_summary": {
                "status": "blocked",
                "next_action": "run_full_training_on_stage7_v3_rd_segment_store",
                "full_training_rerun_recommended_now": True,
            }
        }

        final_readiness = _replace_precision_fields(
            evaluation_report,
            summary,
            {"checkpoint_name": "last_full_val", "path": "artifacts/run/last_full_val_model.pt"},
            precision_readiness,
            {},
            {"status": "exported", "path": "artifacts/run/precision.pt"},
            {"status": "metadata_only_not_evaluated"},
        )
        _apply_test_reporting(evaluation_report, summary, test_observations)

        for holder in [
            {"precision_candidate_readiness": final_readiness},
            evaluation_report,
            summary,
            evaluation_report["checkpoint_selection"],
            evaluation_report["diagnostic_checkpoint_tests"],
        ]:
            self.assertEqual(holder["precision_candidate_readiness"], precision_readiness)
        self.assertEqual(evaluation_report["test_reporting"]["report_scope"], "test")
        self.assertIs(evaluation_report["test_reporting"]["test_used_for_selection"], False)

    def test_markdown_report_includes_diagnostic_checkpoint_table(self) -> None:
        parent_aggregation = _build_stage5_window_aggregation()
        evaluation_report = {
            "summary": {
                "loss": 0.1,
                "accuracy": 0.8,
                "macro_precision": 0.8,
                "macro_recall": 0.8,
                "macro_f1": 0.8,
                "route_accuracy": 1.0,
                "selected_joint_logit_delta_scale": 0.5,
            },
            "label_names": LOW_ALTITUDE_CLASSES,
            "classification_confusion_matrix": [
                [4, 0, 0, 0, 0],
                [0, 4, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 4],
            ],
            "per_class": [
                {"label_name": label_name, "support": 5, "precision": 0.8, "recall": 0.8, "f1": 0.8}
                for label_name in LOW_ALTITUDE_CLASSES
            ],
            "route": {
                "label_names": ["low_altitude_suspect"],
                "confusion_matrix": [[10]],
                "per_route": [{"route_name": "low_altitude_suspect", "support": 10, "correct": 10, "recall": 1.0}],
            },
            "by_source": {},
            "predictions": [],
            "window_evaluation": {
                "index_name": "expert1_window_index.csv",
                "is_window_index": True,
                "parent_aggregation": parent_aggregation,
            },
            "rd_time_sync_metrics": {
                "exact_time_sync": {
                    "support": 4,
                    "accuracy": 0.75,
                    "macro_f1": 0.7000,
                    "traj_macro_f1": 0.6800,
                    "joint_macro_f1": 0.7200,
                    "joint_minus_traj_macro_f1": 0.0200,
                    "rd_macro_f1": 0.6200,
                },
                "proportional_fallback": {
                    "support": 6,
                    "accuracy": 0.80,
                    "macro_f1": 0.7600,
                    "traj_macro_f1": 0.7800,
                    "joint_macro_f1": 0.7500,
                    "joint_minus_traj_macro_f1": -0.0300,
                    "rd_macro_f1": 0.5900,
                },
            },
            "rd_joint_bottleneck_summary": {
                "status": "blocked",
                "primary_bottleneck": "rd_evidence_below_release_and_joint_fusion_negative",
                "next_action": "run_stage6d_rd_joint_bottleneck_diagnostic_before_full_training",
                "full_training_rerun_recommended_now": False,
                "diagnosis_flags": [
                    "joint_on_rd_below_precision_tolerance",
                    "rd_evidence_below_release_gate",
                    "unknown_clutter_rd_worse_than_traj",
                ],
                "metrics": {
                    "with_rd_count": 10,
                    "selected_joint_logit_delta_scale": 0.25,
                    "joint_minus_traj_macro_f1_on_rd": -0.0249,
                    "rd_macro_f1_on_rd": 0.5354,
                    "final_net_gain_rate_on_rd": 0.0023,
                    "final_harm_rate_on_rd": 0.0057,
                    "final_unknown_recall": 0.7024,
                    "final_clutter_recall": 0.5333,
                },
                "release_gaps": {
                    "precision_joint_minus_traj_macro_f1_on_rd_gap": 0.0149,
                    "joint_minus_traj_macro_f1_on_rd_gap": 0.0399,
                    "rd_macro_f1_on_rd_gap": 0.0846,
                    "final_net_gain_rate_on_rd_gap": 0.0127,
                    "final_harm_rate_on_rd_gap": -0.0443,
                },
                "hard_pair_bottlenecks": [
                    {
                        "pair_name": "UnknownTarget__vs__ClutterTarget",
                        "support": 125,
                        "traj_accuracy": 0.6240,
                        "rd_accuracy": 0.3840,
                        "rd_minus_traj_accuracy": -0.2400,
                        "final_accuracy": 0.5600,
                        "final_minus_traj_accuracy": -0.0640,
                        "rd_net_gain_rate": -0.2400,
                        "final_net_gain_rate": -0.0640,
                    }
                ],
                "source_bottlenecks_on_rd": [
                    {
                        "source_name": "cq08_track",
                        "support_on_rd": 869,
                        "traj_macro_f1_on_rd": 0.7585,
                        "joint_macro_f1_on_rd": 0.7299,
                        "joint_minus_traj_macro_f1_on_rd": -0.0287,
                        "rd_macro_f1_on_rd": 0.5772,
                        "final_macro_f1_on_rd": 0.7485,
                        "final_minus_traj_macro_f1_on_rd": -0.0100,
                    }
                ],
                "time_sync_bottlenecks": [
                    {
                        "rd_time_sync_status": "proportional_fallback",
                        "support": 869,
                        "macro_f1": 0.7485,
                        "traj_macro_f1": 0.7585,
                        "joint_macro_f1": 0.7299,
                        "joint_minus_traj_macro_f1": -0.0287,
                        "rd_macro_f1": 0.5772,
                    }
                ],
            },
            "checkpoint_selection": {
                "candidate_status": "invalid_for_release",
                "selected_checkpoint": "artifacts/run/best_model.pt",
                "selected_checkpoint_name": "best_selection",
                "precision_checkpoint": "artifacts/run/precision_candidate_model.pt",
                "precision_checkpoint_name": "last_full_val",
                "precision_checkpoint_export": {
                    "path": "artifacts/run/precision_candidate_model.pt",
                    "source_path": "artifacts/run/last_full_val_model.pt",
                    "source_checkpoint_name": "last_full_val",
                    "recommended_joint_logit_delta_scale": 0.5,
                },
                "precision_checkpoint_validation": {
                    "status": "ok",
                    "path": "artifacts/run/precision_candidate_model.pt",
                    "source_checkpoint_name": "last_full_val",
                    "selected_joint_logit_delta_scale": 0.5,
                    "expected_joint_logit_delta_scale": 0.5,
                    "scale_matches": True,
                    "macro_f1": 0.8150,
                    "expected_macro_f1": 0.8150,
                    "macro_f1_delta": 0.0,
                    "macro_f1_matches": True,
                    "matches_precision_candidate": True,
                },
                "selected_reason": "fallback_v2_selection_checkpoint",
                "selected_checkpoint_eligible": True,
                "release_eligible": False,
                "release_gate_failures": ["joint_minus_traj_macro_f1_on_rd", "final_net_gain_rate_on_rd"],
                "release_gate_metrics": {
                    "release_macro_f1": 0.7907,
                    "release_final_minus_traj_macro_f1": 0.0095,
                    "release_joint_minus_traj_macro_f1_on_rd": -0.0100,
                    "release_final_net_gain_rate_on_rd": -0.0020,
                    "release_final_harm_rate_on_rd": 0.0200,
                    "release_parent_macro_f1": 0.8300,
                    "release_final_unknown_recall": 0.5600,
                    "release_final_clutter_recall": 0.5100,
                    "release_without_rd_accuracy": 0.9000,
                },
                "best_epoch": 14,
            },
            "diagnostic_checkpoint_tests": {
                "selected_checkpoint": "artifacts/run/best_model.pt",
                "selected_checkpoint_name": "best_selection",
                "precision_candidate": {
                    "checkpoint_name": "last_full_val",
                    "epoch": 18,
                    "recommendation_basis": "diagnostic_test_precision_candidate_calibrated",
                    "is_selected_checkpoint": False,
                    "precision_threshold_failures": [],
                    "macro_f1": 0.8150,
                    "traj_macro_f1": 0.8020,
                    "joint_minus_traj_macro_f1_on_rd": -0.0060,
                    "rd_macro_f1_on_rd": 0.5900,
                    "final_net_gain_rate_on_rd": 0.0137,
                    "final_harm_rate_on_rd": 0.0046,
                    "final_unknown_recall": 0.7857,
                    "final_clutter_recall": 0.5500,
                    "parent_mean_logit_macro_f1": 0.8077,
                    "source_balanced_supported_macro_f1": 0.6052,
                    "precision_threshold_gaps": {
                        "precision_candidate_unknown_recall_gap": -0.0857,
                        "precision_candidate_clutter_recall_gap": -0.0500,
                        "precision_candidate_joint_gain_gap": -0.0040,
                    },
                    "recommended_joint_logit_delta_scale": 0.5,
                    "precision_logit_bias_by_label": {"UnknownTarget": 0.0500},
                    "precision_calibration_probe": {
                        "status": "ready",
                        "ready": True,
                        "recommended": {
                            "macro_f1": 0.8150,
                            "final_net_gain_rate_on_rd": 0.0137,
                            "final_harm_rate_on_rd": 0.0046,
                            "final_unknown_recall": 0.7857,
                            "final_clutter_recall": 0.5500,
                            "parent_mean_logit_macro_f1": 0.8077,
                            "precision_logit_bias_by_label": {"UnknownTarget": 0.0500},
                            "precision_threshold_failures": [],
                        },
                        "results": [
                            {
                                "macro_f1": 0.8150,
                                "final_net_gain_rate_on_rd": 0.0137,
                                "final_harm_rate_on_rd": 0.0046,
                                "final_unknown_recall": 0.7857,
                                "final_clutter_recall": 0.5500,
                                "precision_logit_bias_by_label": {"UnknownTarget": 0.0500},
                                "precision_threshold_failures": [],
                            }
                        ],
                    },
                },
                "precision_candidate_delta_vs_selected": {
                    "selected_checkpoint_name": "best_selection",
                    "precision_candidate_checkpoint_name": "last_full_val",
                    "is_same_checkpoint": False,
                    "selection_mismatch": True,
                    "macro_f1_delta": 0.0243,
                    "traj_macro_f1_delta": 0.0208,
                    "joint_minus_traj_macro_f1_on_rd_delta": 0.0040,
                    "rd_macro_f1_on_rd_delta": -0.0400,
                    "final_net_gain_rate_on_rd_delta": -0.0073,
                    "final_harm_rate_on_rd_delta": -0.0154,
                    "final_unknown_recall_delta": 0.2257,
                    "final_clutter_recall_delta": 0.0400,
                    "parent_mean_logit_macro_f1_delta": -0.0223,
                },
                "diagnostic_test_recommendation": {
                    "checkpoint_name": "best_joint_gain",
                    "epoch": 12,
                    "recommendation_basis": "diagnostic_test_release_proxy",
                    "macro_f1": 0.8050,
                    "traj_macro_f1": 0.7900,
                    "joint_minus_traj_macro_f1_on_rd": 0.0200,
                    "rd_macro_f1_on_rd": 0.6200,
                    "final_net_gain_rate_on_rd": 0.0180,
                    "final_harm_rate_on_rd": 0.0200,
                    "final_unknown_recall": 0.5800,
                    "final_clutter_recall": 0.5200,
                    "source_balanced_supported_macro_f1": 0.6400,
                },
                "results": [
                    {
                        "checkpoint_name": "best_selection",
                        "status": "ok",
                        "epoch": 14,
                        "is_selected_checkpoint": True,
                        "selected_joint_logit_delta_scale": 0.5,
                        "macro_f1": 0.7907,
                        "traj_macro_f1": 0.7812,
                        "joint_macro_f1_on_rd": 0.8011,
                        "joint_minus_traj_macro_f1_on_rd": 0.0400,
                        "rd_macro_f1_on_rd": 0.6300,
                        "final_net_gain_rate_on_rd": 0.0210,
                        "final_harm_rate_on_rd": 0.0300,
                        "final_unknown_recall": 0.5600,
                        "final_clutter_recall": 0.5100,
                        "parent_mean_logit_macro_f1": 0.8300,
                        "source_balanced_supported_macro_f1": 0.6230,
                        "joint_logit_delta_probe_best_scale": 0.5,
                        "conservative_precision_scale": 0.5,
                    },
                    {
                        "checkpoint_name": "best_joint_gain",
                        "status": "ok",
                        "epoch": 12,
                        "is_selected_checkpoint": False,
                        "selected_joint_logit_delta_scale": 1.0,
                        "macro_f1": 0.7700,
                        "traj_macro_f1": 0.7600,
                        "joint_macro_f1_on_rd": 0.7900,
                        "joint_minus_traj_macro_f1_on_rd": 0.0200,
                        "rd_macro_f1_on_rd": 0.6000,
                        "final_net_gain_rate_on_rd": 0.0100,
                        "final_harm_rate_on_rd": 0.0200,
                        "final_unknown_recall": 0.4800,
                        "final_clutter_recall": 0.4200,
                        "parent_mean_logit_macro_f1": 0.8100,
                        "source_balanced_supported_macro_f1": 0.6000,
                        "joint_logit_delta_probe_best_scale": 0.25,
                        "conservative_precision_scale": 0.25,
                    },
                ],
            },
            "joint_logit_delta_probe": {
                "selection_metric": "macro_f1_weak_class_guard",
                "selected_scale": 0.5,
                "best_test_scale": 0.5,
                "results": [
                    {
                        "scale": 0.0,
                        "macro_f1": 0.7600,
                        "joint_minus_traj_macro_f1_on_rd": 0.0400,
                        "final_net_gain_rate_on_rd": -0.0100,
                        "final_harm_rate_on_rd": 0.0600,
                        "final_unknown_recall": 0.5000,
                        "final_clutter_recall": 0.4500,
                        "unknown_clutter_hard_pair": {
                            "support": 2,
                            "traj_accuracy": 0.5000,
                            "rd_accuracy": 0.5000,
                            "final_accuracy": 0.5000,
                            "rd_net_gain_rate": 0.0000,
                            "final_net_gain_rate": -0.0100,
                            "UnknownTarget_recall": 0.5000,
                            "ClutterTarget_recall": 0.4500,
                        },
                    },
                    {
                        "scale": 0.5,
                        "macro_f1": 0.7907,
                        "joint_minus_traj_macro_f1_on_rd": 0.0400,
                        "final_net_gain_rate_on_rd": 0.0210,
                        "final_harm_rate_on_rd": 0.0300,
                        "final_unknown_recall": 0.5600,
                        "final_clutter_recall": 0.5100,
                        "unknown_clutter_hard_pair": {
                            "support": 2,
                            "traj_accuracy": 0.5000,
                            "rd_accuracy": 0.5000,
                            "final_accuracy": 0.5000,
                            "rd_net_gain_rate": 0.0000,
                            "final_net_gain_rate": 0.0000,
                            "UnknownTarget_recall": 0.5000,
                            "ClutterTarget_recall": 0.5000,
                        },
                    },
                ],
            },
        }
        diagnostic_tests = evaluation_report["diagnostic_checkpoint_tests"]
        readiness = _precision_candidate_readiness_summary(
            diagnostic_tests["results"],
            diagnostic_tests["precision_candidate"],
        )
        diagnostic_tests["precision_candidate_readiness"] = readiness
        evaluation_report["checkpoint_selection"]["precision_candidate_readiness"] = readiness
        summary = {
            "best_checkpoint": "artifacts/run/best_model.pt",
            "precision_checkpoint": "artifacts/run/precision_candidate_model.pt",
            "precision_checkpoint_name": "last_full_val",
            "precision_checkpoint_export": {
                "path": "artifacts/run/precision_candidate_model.pt",
                "source_path": "artifacts/run/last_full_val_model.pt",
                "source_checkpoint_name": "last_full_val",
                "recommended_joint_logit_delta_scale": 0.5,
            },
            "precision_checkpoint_validation": {
                "status": "ok",
                "path": "artifacts/run/precision_candidate_model.pt",
                "source_checkpoint_name": "last_full_val",
                "selected_joint_logit_delta_scale": 0.5,
                "expected_joint_logit_delta_scale": 0.5,
                "scale_matches": True,
                "macro_f1": 0.8150,
                "expected_macro_f1": 0.8150,
                "macro_f1_delta": 0.0,
                "macro_f1_matches": True,
                "matches_precision_candidate": True,
            },
            "history_file": "artifacts/run/history.json",
            "manifest_file": "artifacts/run/manifest.json",
            "evaluation_report": "artifacts/run/evaluation_report.json",
            "test_metrics": {},
        }

        markdown = build_markdown_report(evaluation_report, summary, "diagnostic-checkpoint-report")

        self.assertIn("## Checkpoint 选择", markdown)
        self.assertIn("- precision_checkpoint: artifacts/run/precision_candidate_model.pt", markdown)
        self.assertIn("- precision_checkpoint_name: last_full_val", markdown)
        self.assertIn("- precision_checkpoint_source: artifacts/run/last_full_val_model.pt", markdown)
        self.assertIn("- precision_checkpoint_joint_logit_delta_scale: 0.5000", markdown)
        self.assertIn("- precision_checkpoint_validation_status: ok", markdown)
        self.assertIn("- precision_checkpoint_validation_scale: 0.5000", markdown)
        self.assertIn("- precision_checkpoint_validation_macro_f1: 0.8150", markdown)
        self.assertIn("- precision_checkpoint_validation_macro_f1_delta: 0.0000", markdown)
        self.assertIn("### Release Gate", markdown)
        self.assertIn("### Precision Candidate", markdown)
        self.assertIn("- checkpoint_name: last_full_val", markdown)
        self.assertIn("- recommended_joint_logit_delta_scale: 0.5000", markdown)
        self.assertIn("- precision_threshold_gaps:", markdown)
        self.assertIn("| precision_candidate_clutter_recall_gap | -0.0500 |", markdown)
        self.assertIn("| precision_candidate_unknown_recall_gap | -0.0857 |", markdown)
        self.assertIn("| precision_candidate_joint_gain_gap | -0.0040 |", markdown)
        self.assertIn("### Precision Calibration Probe", markdown)
        self.assertIn("- recommended_precision_logit_bias: UnknownTarget=0.0500", markdown)
        self.assertIn("selection_mismatch: precision candidate `last_full_val` beats selected `best_selection`", markdown)
        self.assertIn("| 0.8150 | 0.8020 | -0.0060 | 0.5900 | 0.0137 |", markdown)
        self.assertIn("| macro_f1_delta | 0.0243 |", markdown)
        self.assertIn("| rd_macro_f1_on_rd_delta |", markdown)
        self.assertIn("### Precision Readiness", markdown)
        self.assertIn("- existing_scale_ready: False", markdown)
        self.assertIn("- full_training_rerun_recommended_now: False", markdown)
        self.assertIn("### RD/Joint Bottleneck", markdown)
        self.assertIn("- primary_bottleneck: rd_evidence_below_release_and_joint_fusion_negative", markdown)
        self.assertIn("- next_action: run_stage6d_rd_joint_bottleneck_diagnostic_before_full_training", markdown)
        self.assertIn("| precision_joint_minus_traj_macro_f1_on_rd_gap | 0.0149 |", markdown)
        self.assertIn("| UnknownTarget__vs__ClutterTarget | 125 | 0.6240 | 0.3840 | -0.2400 |", markdown)
        self.assertIn("| cq08_track | 869 | 0.7585 | 0.7299 | -0.0287 | 0.5772 |", markdown)
        self.assertIn("| proportional_fallback | 869 | 0.7485 | 0.7585 | 0.7299 | -0.0287 | 0.5772 |", markdown)
        self.assertIn("### Checkpoint Precision View", markdown)
        self.assertIn("| best_selection | 0 | 0.5000 | 14 | 0.7907 |", markdown)
        self.assertIn("### Diagnostic Test Recommendation", markdown)
        self.assertIn("- checkpoint_name: best_joint_gain", markdown)
        self.assertIn("| 0.8050 | 0.7900 | 0.0200 | 0.6200 | 0.0180 |", markdown)
        self.assertIn("- selected_checkpoint_name: best_selection", markdown)
        self.assertIn("- release_eligible: False", markdown)
        self.assertIn("- release_gate_failures: joint_minus_traj_macro_f1_on_rd, final_net_gain_rate_on_rd", markdown)
        self.assertIn("## 训练产物", markdown)
        self.assertIn("- precision_checkpoint: artifacts/run/precision_candidate_model.pt", markdown)
        self.assertIn("### Prediction Flip Diagnostics", markdown)
        self.assertIn("### Parent Weak-Class Recall", markdown)
        self.assertIn("### Unknown/Clutter hard-pair rescue/harm", markdown)
        self.assertIn("### Parent Confusion Matrices", markdown)
        self.assertIn("## RD Time Sync Diagnostics", markdown)
        self.assertIn("| exact_time_sync | 4 | 0.7000 | 0.6800 | 0.7200 | 0.0200 | 0.6200 | 0.7500 |", markdown)
        self.assertIn("### Checkpoint 对比", markdown)
        self.assertIn("### Joint Delta Probe", markdown)
        self.assertIn("| best_selection | 1 | 0.5000 | 0.5000 | 0.5000 | 14 | 0.7907 |", markdown)
        self.assertIn("| best_joint_gain | 0 | 1.0000 | 0.2500 | 0.2500 | 12 | 0.7700 |", markdown)
        self.assertIn("| 0.5000 | 1 | 1 | 0.7907 | 0.0400 | 0.0210 | 0.0300 | 0.5600 | 0.5100 |", markdown)
        self.assertIn("### Joint Delta Hard Pair", markdown)
        self.assertIn("| 0.5000 | 1 | 1 | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |", markdown)
        self.assertIn("flipped_parent_count: 2 / 2", markdown)
        self.assertIn("window -> parent mean-logit", markdown)
        self.assertIn("stable decision", markdown)

    def test_system_loader_restores_precision_checkpoint_joint_scale(self) -> None:
        class TinyLowAltitudeExpert(torch.nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 2)
                self.precision_logit_bias = None

            def set_precision_logit_bias(self, bias: object) -> None:
                self.precision_logit_bias = [float(item) for item in bias]  # type: ignore[arg-type]

        low_altitude_expert = TinyLowAltitudeExpert()
        low_altitude_expert.joint_logit_delta_scale = 1.0
        model = SimpleNamespace(low_altitude_expert=low_altitude_expert)
        checkpoint_path = Path("tmp") / "test_precision_candidate_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path.unlink(missing_ok=True)
            torch.save(
                {
                    "model_state_dict": low_altitude_expert.state_dict(),
                    "selected_joint_logit_delta_scale": 0.25,
                    "precision_logit_bias": [0.0, 0.0, 0.0, 0.05, 0.0],
                    "config": {"joint_logit_delta_scale": 0.25},
                },
                checkpoint_path,
            )
            info = _load_expert1_bundle(model, str(checkpoint_path), torch.device("cpu"))
        finally:
            checkpoint_path.unlink(missing_ok=True)

        self.assertAlmostEqual(float(model.low_altitude_expert.joint_logit_delta_scale), 0.25)
        self.assertAlmostEqual(float(info["selected_joint_logit_delta_scale"]), 0.25)
        self.assertEqual(model.low_altitude_expert.precision_logit_bias, [0.0, 0.0, 0.0, 0.05, 0.0])
        self.assertEqual(info["precision_logit_bias"], [0.0, 0.0, 0.0, 0.05, 0.0])

    def test_system_loader_rejects_unauthorized_checkpoint_bias(self) -> None:
        class TinyLowAltitudeExpert(torch.nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 2)
                self.precision_logit_bias = None

            def set_precision_logit_bias(self, bias: object) -> None:
                self.precision_logit_bias = [float(item) for item in bias]  # type: ignore[arg-type]

        low_altitude_expert = TinyLowAltitudeExpert()
        model = SimpleNamespace(low_altitude_expert=low_altitude_expert)
        checkpoint_path = Path("tmp") / "test_unauthorized_precision_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path.unlink(missing_ok=True)
            torch.save(
                {
                    "model_state_dict": low_altitude_expert.state_dict(),
                    "precision_logit_bias": [9.0] * len(LOW_ALTITUDE_CLASSES),
                    "config": {"precision_logit_bias": [0.0, 0.0, 0.0, 0.05, 0.15]},
                },
                checkpoint_path,
            )
            info = _load_expert1_bundle(model, str(checkpoint_path), torch.device("cpu"))
        finally:
            checkpoint_path.unlink(missing_ok=True)

        expected_bias = [0.0] * len(LOW_ALTITUDE_CLASSES)
        self.assertEqual(model.low_altitude_expert.precision_logit_bias, expected_bias)
        self.assertEqual(info["precision_logit_bias"], expected_bias)


if __name__ == "__main__":
    unittest.main(verbosity=2)
