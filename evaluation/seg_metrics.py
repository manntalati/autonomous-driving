from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch


class ConfusionMatrixMeter:
    """
    Streaming confusion matrix for mIoU. Accumulates over batches, then computes IoU per class.
    Why a meter (not a one-shot fn): val set is too big to hold every pred mask in memory.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        """
        Args: num_classes — number of seg classes; ignore_index — pixel label to skip.
        Stores a (C, C) int64 matrix on CPU.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Zero out the confusion matrix."""
        self.cm.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
          preds — (B, H, W) long predicted class IDs (argmax of logits).
          targets — (B, H, W) long GT class IDs.
        Pipeline:
          1. Flatten both to 1D numpy.
          2. Mask out ignore_index pixels.
          3. Increment cm via np.bincount on (gt * num_classes + pred) then reshape.
        """
        preds = preds.flatten()
        targets = targets.flatten()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        valid = (targets != self.ignore_index)
        preds = preds[valid]
        targets = targets[valid]
        hist = np.bincount(
            targets * self.num_classes + preds,
            minlength=self.num_classes ** 2,
        )
        self.cm += hist.reshape(self.num_classes, self.num_classes)

    def iou_per_class(self) -> np.ndarray:
        """
        Returns: (C,) float array of IoU per class. IoU_c = TP / (TP + FP + FN) for class c.
        Notes:
          - TP_c = cm[c, c].
          - FN_c = cm[c, :].sum() - TP_c.
          - FP_c = cm[:, c].sum() - TP_c.
          - Set NaN for classes with zero union (absent in both pred and GT).
        """
        tp = np.diag(self.cm).astype(np.float64)
        fn = self.cm.sum(axis=1) - tp
        fp = self.cm.sum(axis=0) - tp
        union = tp + fp + fn
        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        present = union > 0
        iou[present] = tp[present] / union[present]
        return iou

    def miou(self) -> float:
        """Mean IoU over classes, ignoring NaNs."""
        return float(np.nanmean(self.iou_per_class()))


def compute_miou(preds: List[np.ndarray], targets: List[np.ndarray], num_classes: int, ignore_index: int = 255) -> Tuple[float, np.ndarray]:
    """
    Convenience wrapper: builds a ConfusionMatrixMeter, ingests lists of arrays, returns (miou, per_class_iou).
    """
    meter = ConfusionMatrixMeter(num_classes=num_classes, ignore_index=ignore_index)
    for pred, target in zip(preds, targets):
        meter.update(pred, target)
    return meter.miou(), meter.iou_per_class()
