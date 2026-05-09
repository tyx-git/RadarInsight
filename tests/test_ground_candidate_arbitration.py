from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.core.forwardlayer.labels import DEFAULT_VOCAB
from src.core.router import ROUTE_NAMES
from src.core.system_config import SystemTrainConfig
from src.core.system_trainer import _project_to_global_logits


def _output(route_weights: torch.Tensor) -> SimpleNamespace:
    batch_size = int(route_weights.shape[0])
    return SimpleNamespace(
        routing=SimpleNamespace(
            route_weights=route_weights,
            route_indices=route_weights.argmax(dim=-1),
            override_mask=torch.zeros(batch_size, dtype=torch.bool),
        ),
        low_altitude_logits=torch.tensor([[0.0, -2.0, -2.0, -2.0, -2.0]] * batch_size, dtype=torch.float32),
        low_altitude_detail=SimpleNamespace(
            traj_cosine=torch.full((batch_size, 5), 0.1),
        ),
        conventional_air_logits=torch.tensor([[0.0, -1.0]] * batch_size, dtype=torch.float32),
        conventional_air_detail=SimpleNamespace(auxiliary={}),
        high_altitude_logits=torch.tensor([[0.0, -1.0]] * batch_size, dtype=torch.float32),
        ground_logits=torch.tensor([[0.0, 1.8, 0.2]] * batch_size, dtype=torch.float32),
        ground_detail=SimpleNamespace(
            auxiliary={
                "groundness_probability": torch.full((batch_size,), 0.95),
                "groundness_threshold": torch.full((batch_size,), 0.50),
            }
        ),
    )


class GroundCandidateArbitrationTest(unittest.TestCase):
    def test_high_confidence_ground_candidate_can_recover_when_router_not_opposed(self) -> None:
        ground_route = ROUTE_NAMES.index("ground_target")
        route_weights = torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=torch.float32)
        labels = torch.tensor([DEFAULT_VOCAB.label_to_index["CarTarget"]])
        config = SystemTrainConfig(
            fusion_mode="soft_log_weighted",
            groundness_protection_enabled=True,
            groundness_confidence_tau_high=0.10,
            ground_route_gate_floor=0.10,
            ground_candidate_margin_tau=0.25,
            aircraft_protection_enabled=False,
        )
        logits, diagnostics = _project_to_global_logits(
            labels,
            _output(route_weights),
            len(DEFAULT_VOCAB.labels),
            torch.device("cpu"),
            config,
            return_diagnostics=True,
        )
        pred_label = DEFAULT_VOCAB.decode(int(logits.argmax(dim=-1).item()))
        self.assertEqual(pred_label, "CarTarget")
        self.assertTrue(bool(diagnostics["ground_candidate_high_confidence"][0].item()))
        self.assertTrue(bool(diagnostics["ground_candidate_changed_prediction"][0].item()))
        self.assertTrue(bool(diagnostics["ground_recovered"][0].item()))
        self.assertFalse(bool(diagnostics["ground_false_positive_introduced"][0].item()))

    def test_router_strongly_against_blocks_ground_candidate_protection(self) -> None:
        ground_route = ROUTE_NAMES.index("ground_target")
        route_weights = torch.tensor([[0.32, 0.32, 0.34, 0.02]], dtype=torch.float32)
        labels = torch.tensor([DEFAULT_VOCAB.label_to_index["CarTarget"]])
        config = SystemTrainConfig(
            fusion_mode="soft_log_weighted",
            groundness_protection_enabled=True,
            groundness_confidence_tau_high=0.10,
            ground_route_gate_floor=0.10,
            ground_candidate_margin_tau=0.25,
            aircraft_protection_enabled=False,
        )
        _, diagnostics = _project_to_global_logits(
            labels,
            _output(route_weights),
            len(DEFAULT_VOCAB.labels),
            torch.device("cpu"),
            config,
            return_diagnostics=True,
        )
        self.assertFalse(bool(diagnostics["ground_route_not_opposed"][0].item()))
        self.assertFalse(bool(diagnostics["ground_candidate_high_confidence"][0].item()))
        self.assertFalse(bool(diagnostics["ground_candidate_changed_prediction"][0].item()))


if __name__ == "__main__":
    unittest.main()
