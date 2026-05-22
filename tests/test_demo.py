"""
Unit tests for Phase 7 demo utilities — BEV detection decode and the
segmentation / BEV visualizers.

Random tensors only — no nuScenes data, GPU, or checkpoints. (The full
PerceptionPipeline needs checkpoints, so it is exercised by demo/benchmark.py,
not here.)
"""

import numpy as np
import pytest
import torch

from models.bev.bev_detector import decode_bev_detections
from utils.visualize import overlay_segmentation, draw_bev

XB = (0.0, 51.2, 0.8)
YB = (-25.6, 25.6, 0.8)


class TestDecodeBEVDetections:

    def test_single_peak_decoded(self):
        """A clean heatmap peak decodes to the expected BEV box."""
        hm = torch.zeros(3, 64, 64)
        hm[0, 30, 30] = 0.9
        reg = torch.zeros(6, 64, 64)
        reg[2, 30, 30] = 4.0   # length
        reg[3, 30, 30] = 2.0   # width
        reg[5, 30, 30] = 1.0   # cos(yaw) = 1 → yaw 0
        boxes, scores, labels = decode_bev_detections(hm, reg, XB, YB)
        assert boxes.shape == (1, 5)
        x, y, length, width, yaw = boxes[0].tolist()
        assert x == pytest.approx(24.0)      # (30 + 0)·0.8 + 0.0
        assert y == pytest.approx(-1.6)      # (30 + 0)·0.8 - 25.6
        assert (length, width, yaw) == pytest.approx((4.0, 2.0, 0.0))
        assert scores[0] == pytest.approx(0.9)
        assert labels[0].item() == 0

    def test_empty_heatmap_returns_no_boxes(self):
        boxes, scores, labels = decode_bev_detections(
            torch.zeros(3, 64, 64), torch.zeros(6, 64, 64), XB, YB)
        assert boxes.shape == (0, 5) and scores.numel() == 0 and labels.numel() == 0

    def test_score_threshold_filters_weak_peaks(self):
        hm = torch.zeros(3, 64, 64)
        hm[1, 10, 10] = 0.2   # below the default 0.3 threshold
        boxes, _, _ = decode_bev_detections(hm, torch.zeros(6, 64, 64), XB, YB)
        assert boxes.shape[0] == 0

    def test_max_detections_caps_output(self):
        hm = torch.zeros(3, 64, 64)
        # peaks 2 cells apart so each survives the 3×3 local-max test
        for i in range(10):
            hm[0, 2 * i, 2 * i] = 0.5 + 0.01 * i
        boxes, _, _ = decode_bev_detections(
            hm, torch.zeros(6, 64, 64), XB, YB, max_detections=4)
        assert boxes.shape[0] == 4


class TestOverlaySegmentation:

    def test_shape_and_dtype_preserved(self):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        mask = np.random.randint(0, 5, (48, 64))
        out = overlay_segmentation(img, mask, alpha=0.5)
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_background_pixels_untouched(self):
        """Class-0 (background) pixels must be left exactly as the input."""
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.int64)   # all background
        out = overlay_segmentation(img, mask, alpha=0.5)
        assert np.array_equal(out, img)

    def test_foreground_pixels_changed(self):
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.int64)    # all drivable
        out = overlay_segmentation(img, mask, alpha=0.5)
        assert not np.array_equal(out, img)


class TestDrawBEV:

    def test_output_shape(self):
        boxes = torch.tensor([[24.0, 0.0, 4.0, 2.0, 0.0]])
        canvas = draw_bev(boxes, torch.tensor([0.9]), torch.tensor([0]), XB, YB, canvas_px=400)
        assert canvas.shape == (400, 400, 3) and canvas.dtype == np.uint8

    def test_empty_boxes_returns_canvas(self):
        canvas = draw_bev(torch.zeros(0, 5), torch.zeros(0), torch.zeros(0, dtype=torch.long),
                          XB, YB, canvas_px=300)
        assert canvas.shape == (300, 300, 3)

    def test_seg_background(self):
        """A BEV semantic map renders as a (non-black) coloured background."""
        seg = np.random.randint(0, 5, (64, 64))   # 64×64 grid matches XB/YB
        canvas = draw_bev(torch.zeros(0, 5), torch.zeros(0), torch.zeros(0, dtype=torch.long),
                          XB, YB, seg=seg, canvas_px=300)
        assert canvas.shape == (300, 300, 3)
        assert canvas.sum() > 0                   # the seg map filled the panel
