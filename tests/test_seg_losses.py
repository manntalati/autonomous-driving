"""
Unit tests for models/segmentation/losses.py — Dice + CrossEntropy segmentation loss.

Verify:
  - dice_loss is a scalar in [0, 1], differentiable, ~0 for a perfect prediction
  - ignore_index pixels are excluded from the Dice computation
  - SegmentationLoss returns (total, log_dict) with the right keys
  - total == ce_weight * ce + dice_weight * dice
  - class_weights path runs

No nuScenes data or GPU required.
"""

import pytest
import torch

from models.segmentation.losses import dice_loss, SegmentationLoss

B, C, H, W = 2, 5, 16, 24


def _random_logits():
    return torch.randn(B, C, H, W, requires_grad=True)


def _random_targets():
    return torch.randint(0, C, (B, H, W))


def _perfect_logits(targets):
    """Logits that, after softmax, are ~one-hot on the GT class."""
    logits = torch.zeros(B, C, H, W)
    logits.scatter_(1, targets.unsqueeze(1), 20.0)
    return logits


class TestDiceLoss:

    def test_returns_scalar(self):
        loss = dice_loss(_random_logits(), _random_targets())
        assert loss.dim() == 0

    def test_in_unit_range(self):
        loss = dice_loss(_random_logits(), _random_targets())
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_prediction_near_zero(self):
        targets = _random_targets()
        loss = dice_loss(_perfect_logits(targets), targets)
        assert loss.item() < 0.01

    def test_differentiable(self):
        logits = _random_logits()
        dice_loss(logits, _random_targets()).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_ignore_index_excludes_pixels(self):
        """A target that is all ignore_index leaves nothing to score → loss 0."""
        targets = torch.full((B, H, W), 255)
        loss = dice_loss(_random_logits(), targets, ignore_index=255)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_ignore_runs(self):
        """A mix of valid and ignored pixels must still produce a finite loss."""
        targets = _random_targets()
        targets[0, :H // 2, :] = 255
        loss = dice_loss(_random_logits(), targets, ignore_index=255)
        assert torch.isfinite(loss)


class TestSegmentationLoss:

    def test_returns_tuple_with_log_dict(self):
        crit = SegmentationLoss(num_classes=C)
        total, log = crit(_random_logits(), _random_targets())
        assert isinstance(total, torch.Tensor) and total.dim() == 0
        assert set(log.keys()) == {"loss", "ce_loss", "dice_loss"}

    def test_total_is_weighted_sum(self):
        """total must equal ce_weight*ce + dice_weight*dice from the log dict."""
        crit = SegmentationLoss(num_classes=C, ce_weight=2.0, dice_weight=0.5)
        total, log = crit(_random_logits(), _random_targets())
        expected = 2.0 * log["ce_loss"] + 0.5 * log["dice_loss"]
        assert total.item() == pytest.approx(expected, rel=1e-5)

    def test_backward_flows(self):
        crit = SegmentationLoss(num_classes=C)
        logits = _random_logits()
        total, _ = crit(logits, _random_targets())
        total.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_class_weights_path(self):
        """Passing class_weights must run and produce a finite loss."""
        weights = torch.tensor([0.5, 1.0, 1.0, 2.0, 1.5])
        crit = SegmentationLoss(num_classes=C, class_weights=weights)
        total, _ = crit(_random_logits(), _random_targets())
        assert torch.isfinite(total)

    def test_class_weights_change_loss(self):
        """Weighted and unweighted CE should differ for a non-uniform target."""
        torch.manual_seed(0)
        logits = _random_logits()
        targets = _random_targets()
        plain = SegmentationLoss(num_classes=C)(logits, targets)[1]["ce_loss"]
        weighted = SegmentationLoss(
            num_classes=C, class_weights=torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0])
        )(logits, targets)[1]["ce_loss"]
        assert plain != pytest.approx(weighted, rel=1e-4)