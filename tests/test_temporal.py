"""
Unit tests for Phase 6 — temporal cross-attention, the temporal detector,
and the flicker (temporal-consistency) metric.

Random tensors only — no nuScenes data, GPU, or weight downloads.
(The sequence dataset needs nuScenes, so it is exercised by the smoke test,
not here — consistent with the offline-only test convention.)
"""

import pytest
import torch

from models.temporal.temporal_attention import TemporalCrossAttention
from models.temporal.train_temporal import build_temporal_detector
from evaluation.temporal_metrics import compute_flicker_rate


def _detector_cfg():
    return dict(
        num_classes=3, num_anchors=3, scales=[128, 64, 32],
        aspect_ratios=[0.5, 1.0, 2.0], strides=[32, 16, 8],
        seq_len=3, temporal_heads=8, pretrained=False,
    )


class TestTemporalCrossAttention:

    def test_shape_preserving(self):
        """Fusing the current frame with its past leaves the shape unchanged."""
        attn = TemporalCrossAttention(embed_dim=64, num_heads=8, num_past_frames=2)
        cur = torch.randn(2, 64, 7, 10)
        past = [torch.randn(2, 64, 7, 10), torch.randn(2, 64, 7, 10)]
        assert attn(cur, past).shape == cur.shape

    def test_differentiable(self):
        attn = TemporalCrossAttention(64, 8, num_past_frames=2)
        cur = torch.randn(1, 64, 5, 5, requires_grad=True)
        past = [torch.randn(1, 64, 5, 5), torch.randn(1, 64, 5, 5)]
        attn(cur, past).sum().backward()
        assert cur.grad is not None and torch.isfinite(cur.grad).all()

    def test_rejects_indivisible_heads(self):
        with pytest.raises(AssertionError):
            TemporalCrossAttention(embed_dim=64, num_heads=7)

    def test_temporal_embed_shape(self):
        """One learned embedding row per past frame."""
        attn = TemporalCrossAttention(64, 8, num_past_frames=2)
        assert attn.temporal_embed.shape == (2, 64)

    def test_single_past_frame(self):
        attn = TemporalCrossAttention(64, 8, num_past_frames=1)
        cur = torch.randn(1, 64, 5, 5)
        assert attn(cur, [torch.randn(1, 64, 5, 5)]).shape == cur.shape


class TestTemporalDetector:

    def test_train_mode_returns_raw(self):
        """In train mode → (cls_logits, bbox_deltas, anchors)."""
        model = build_temporal_detector(_detector_cfg()).train()
        out = model(torch.randn(2, 3, 3, 128, 128))
        assert len(out) == 3
        cls_logits, bbox_deltas, _ = out
        assert len(cls_logits) == 3 and len(bbox_deltas) == 3  # one per FPN level

    def test_eval_mode_postprocesses(self):
        """In eval mode → per-image (boxes, scores, labels)."""
        model = build_temporal_detector(_detector_cfg()).eval()
        boxes, scores, labels = model(torch.randn(1, 3, 3, 128, 128))
        assert len(boxes) == 1 and len(scores) == 1 and len(labels) == 1

    def test_backward_reaches_temporal_module(self):
        """Gradients must flow into the temporal cross-attention parameters."""
        model = build_temporal_detector(_detector_cfg()).train()
        cls_logits, bbox_deltas, _ = model(torch.randn(1, 3, 3, 128, 128))
        loss = sum(c.sum() for c in cls_logits) + sum(b.sum() for b in bbox_deltas)
        loss.backward()
        assert any(p.grad is not None for p in model.temporal_attn.parameters())


class TestComputeFlickerRate:

    BOX = [[0.0, 0.0, 10.0, 10.0]]

    def test_flicker_when_missed_in_middle(self):
        """Detected at t-1 and t+1, missed at t → flicker rate 1.0."""
        rate = compute_flicker_rate(
            seq_pred_boxes=[self.BOX, [], self.BOX],          # missed in frame 1
            seq_gt_boxes=[self.BOX, self.BOX, self.BOX],
            seq_gt_instances=[["carA"], ["carA"], ["carA"]],
        )
        assert rate == pytest.approx(1.0)

    def test_no_flicker_when_consistently_detected(self):
        rate = compute_flicker_rate(
            seq_pred_boxes=[self.BOX, self.BOX, self.BOX],
            seq_gt_boxes=[self.BOX, self.BOX, self.BOX],
            seq_gt_instances=[["carA"], ["carA"], ["carA"]],
        )
        assert rate == pytest.approx(0.0)

    def test_no_candidate_triples_returns_zero(self):
        """An object that never appears in 3 consecutive frames yields no triples."""
        rate = compute_flicker_rate(
            seq_pred_boxes=[self.BOX, [], []],
            seq_gt_boxes=[self.BOX, [], []],
            seq_gt_instances=[["carA"], [], []],
        )
        assert rate == 0.0

    def test_missed_neighbour_is_not_a_candidate(self):
        """If t-1 is itself a miss, frame t is not a candidate triple → no flicker."""
        rate = compute_flicker_rate(
            seq_pred_boxes=[[], [], self.BOX],   # only frame 2 detected
            seq_gt_boxes=[self.BOX, self.BOX, self.BOX],
            seq_gt_instances=[["carA"], ["carA"], ["carA"]],
        )
        assert rate == 0.0
