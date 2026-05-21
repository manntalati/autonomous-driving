"""
Unit tests for evaluation/seg_metrics.py — streaming mIoU via a confusion matrix.

Verify:
  - perfect prediction → mIoU 1.0
  - a hand-built confusion case yields the analytically-known IoU
  - ignore_index pixels never enter the matrix
  - classes absent from both pred and GT are NaN and skipped by miou()
  - the matrix accumulates across update() calls and reset() clears it
  - compute_miou ingests a list of arrays

No nuScenes data or GPU required.
"""

import numpy as np
import pytest
import torch

from evaluation.seg_metrics import ConfusionMatrixMeter, compute_miou


class TestConfusionMatrixMeter:

    def test_perfect_prediction(self):
        meter = ConfusionMatrixMeter(num_classes=5)
        t = torch.randint(0, 5, (2, 8, 8))
        meter.update(t.clone(), t.clone())
        assert meter.miou() == pytest.approx(1.0)
        assert np.allclose(meter.iou_per_class(), 1.0)

    def test_known_confusion_case(self):
        """targets=[0,0,1,1], preds=[0,1,1,1] → cm=[[1,1],[0,2]].

        class 0: TP=1 FP=0 FN=1 → IoU 0.5
        class 1: TP=2 FP=1 FN=0 → IoU 2/3
        """
        meter = ConfusionMatrixMeter(num_classes=2)
        targets = torch.tensor([[[0, 0], [1, 1]]])  # (1, 2, 2)
        preds = torch.tensor([[[0, 1], [1, 1]]])
        meter.update(preds, targets)
        iou = meter.iou_per_class()
        assert iou[0] == pytest.approx(0.5)
        assert iou[1] == pytest.approx(2.0 / 3.0)
        assert meter.miou() == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)

    def test_ignore_index_excluded(self):
        """An all-ignore batch must leave the confusion matrix empty."""
        meter = ConfusionMatrixMeter(num_classes=5, ignore_index=255)
        targets = torch.full((1, 8, 8), 255)
        preds = torch.randint(0, 5, (1, 8, 8))
        meter.update(preds, targets)
        assert meter.cm.sum() == 0

    def test_absent_class_is_nan_and_skipped(self):
        """A class absent from both pred and GT → NaN IoU, ignored by miou()."""
        meter = ConfusionMatrixMeter(num_classes=5)
        t = torch.zeros(1, 8, 8, dtype=torch.long)  # only class 0 appears
        meter.update(t.clone(), t.clone())
        iou = meter.iou_per_class()
        assert iou[0] == pytest.approx(1.0)
        assert np.isnan(iou[1:]).all()
        assert meter.miou() == pytest.approx(1.0)  # NaNs skipped

    def test_matrix_shape_with_only_low_classes(self):
        """update() with only class 0 present must still yield a (C, C) matrix.

        Guards the np.bincount minlength fix — without it the matrix would
        be too small to reshape.
        """
        meter = ConfusionMatrixMeter(num_classes=5)
        t = torch.zeros(1, 4, 4, dtype=torch.long)
        meter.update(t.clone(), t.clone())
        assert meter.cm.shape == (5, 5)

    def test_accumulates_across_updates(self):
        meter = ConfusionMatrixMeter(num_classes=3)
        t = torch.randint(0, 3, (1, 6, 6))
        meter.update(t.clone(), t.clone())
        first = meter.cm.sum()
        meter.update(t.clone(), t.clone())
        assert meter.cm.sum() == 2 * first

    def test_reset_clears_matrix(self):
        meter = ConfusionMatrixMeter(num_classes=3)
        t = torch.randint(0, 3, (1, 6, 6))
        meter.update(t.clone(), t.clone())
        meter.reset()
        assert meter.cm.sum() == 0

    def test_miou_returns_python_float(self):
        meter = ConfusionMatrixMeter(num_classes=3)
        t = torch.randint(0, 3, (1, 6, 6))
        meter.update(t.clone(), t.clone())
        assert isinstance(meter.miou(), float)


class TestComputeMiou:

    def test_list_of_arrays(self):
        """compute_miou must ingest a list of numpy arrays and match a perfect score."""
        a = np.array([[0, 1], [2, 3]])
        miou, iou = compute_miou([a], [a.copy()], num_classes=5)
        assert miou == pytest.approx(1.0)
        assert iou.shape == (5,)

    def test_multiple_arrays_accumulate(self):
        preds = [np.array([[0, 0]]), np.array([[1, 1]])]
        targets = [np.array([[0, 0]]), np.array([[1, 1]])]
        miou, _ = compute_miou(preds, targets, num_classes=2)
        assert miou == pytest.approx(1.0)