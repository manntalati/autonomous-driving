"""
Unit tests for the Phase 5 BEV detector — encoder, head, full detector,
and the BEV detection targets + loss.

Random tensors only — no nuScenes data, GPU, or weight downloads.
"""

import pytest
import torch

from models.bev.bev_detector import BEVEncoder, BEVDetectionHead, BEVDetector
from models.bev.losses import encode_bev_targets, BEVDetectionLoss


class TestBEVEncoder:

    def test_shape_preserving(self):
        x = torch.randn(2, 16, 8, 8)
        assert BEVEncoder(16)(x).shape == x.shape


class TestBEVDetectionHead:

    def test_output_shapes(self):
        hm, reg = BEVDetectionHead(in_channels=16, num_classes=3)(torch.randn(2, 16, 8, 8))
        assert hm.shape == (2, 3, 8, 8)
        assert reg.shape == (2, 6, 8, 8)

    def test_heatmap_in_unit_range(self):
        """Heatmap is a sigmoid → strictly within [0, 1]."""
        hm, _ = BEVDetectionHead(16, 3)(torch.randn(2, 16, 8, 8))
        assert hm.min() >= 0.0 and hm.max() <= 1.0


class TestBEVDetector:

    @pytest.fixture
    def model(self):
        return BEVDetector(num_classes=3, image_size=(64, 64))

    def test_forward_shapes(self, model):
        """Surround input (B, N, 3, H, W) → heatmap (B,3,X,Y) + regression (B,6,X,Y)."""
        B, N = 2, 2
        hm, reg = model(torch.randn(B, N, 3, 64, 64),
                        torch.eye(3).repeat(B, N, 1, 1), torch.eye(4).repeat(B, N, 1, 1))
        assert hm.shape == (B, 3, 64, 64)
        assert reg.shape == (B, 6, 64, 64)

    def test_backward(self, model):
        hm, reg = model(torch.randn(1, 1, 3, 64, 64),
                        torch.eye(3).repeat(1, 1, 1, 1), torch.eye(4).repeat(1, 1, 1, 1))
        (hm.sum() + reg.sum()).backward()
        assert all(p.grad is not None for p in model.head.parameters())


class TestEncodeBEVTargets:

    def test_output_shapes(self):
        hm, reg, mask = encode_bev_targets(
            torch.tensor([[3.0, 5.0, 2.0, 2.0, 0.0]]), torch.tensor([1]),
            num_classes=3, xbound=(0, 8, 2), ybound=(0, 8, 2),
        )
        assert hm.shape == (3, 4, 4) and reg.shape == (6, 4, 4) and mask.shape == (4, 4)

    def test_box_lands_in_expected_cell(self):
        """x=3 → cx 1.5 → cell 1 ; y=5 → cy 2.5 → cell 2."""
        hm, reg, mask = encode_bev_targets(
            torch.tensor([[3.0, 5.0, 2.0, 2.0, 0.0]]), torch.tensor([1]),
            num_classes=3, xbound=(0, 8, 2), ybound=(0, 8, 2),
        )
        assert hm[1, 1, 2] == pytest.approx(1.0)   # Gaussian peak at the centre cell
        assert mask[1, 2] == 1.0
        assert reg[0, 1, 2] == pytest.approx(0.5)  # sub-cell offset x
        assert reg[1, 1, 2] == pytest.approx(0.5)  # sub-cell offset y
        assert reg[2, 1, 2] == pytest.approx(2.0)  # length
        assert reg[5, 1, 2] == pytest.approx(1.0)  # cos(yaw=0)

    def test_empty_boxes(self):
        hm, reg, mask = encode_bev_targets(
            torch.zeros(0, 5), torch.zeros(0, dtype=torch.long),
            num_classes=3, xbound=(0, 8, 2), ybound=(0, 8, 2),
        )
        assert hm.sum() == 0 and mask.sum() == 0

    def test_box_outside_grid_dropped(self):
        _, _, mask = encode_bev_targets(
            torch.tensor([[100.0, 100.0, 2.0, 2.0, 0.0]]), torch.tensor([0]),
            num_classes=3, xbound=(0, 8, 2), ybound=(0, 8, 2),
        )
        assert mask.sum() == 0


class TestBEVDetectionLoss:

    def _targets(self):
        return [
            {"boxes": torch.tensor([[3.0, 5.0, 2.0, 2.0, 0.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)},
        ]

    def test_returns_scalar_and_log(self):
        crit = BEVDetectionLoss(3, (0, 8, 2), (0, 8, 2))
        total, log = crit(torch.rand(2, 3, 4, 4), torch.randn(2, 6, 4, 4), self._targets())
        assert total.dim() == 0
        assert set(log.keys()) == {"loss", "hm_loss", "reg_loss"}

    def test_differentiable(self):
        crit = BEVDetectionLoss(3, (0, 8, 2), (0, 8, 2))
        hm = torch.rand(1, 3, 4, 4, requires_grad=True)
        reg = torch.randn(1, 6, 4, 4, requires_grad=True)
        targets = [{"boxes": torch.tensor([[3.0, 5.0, 2.0, 2.0, 0.0]]), "labels": torch.tensor([1])}]
        total, _ = crit(hm, reg, targets)
        total.backward()
        assert hm.grad is not None and reg.grad is not None

    def test_empty_targets_finite(self):
        """A frame with no GT objects must still give a finite loss."""
        crit = BEVDetectionLoss(3, (0, 8, 2), (0, 8, 2))
        targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
        total, _ = crit(torch.rand(1, 3, 4, 4), torch.randn(1, 6, 4, 4), targets)
        assert torch.isfinite(total)
