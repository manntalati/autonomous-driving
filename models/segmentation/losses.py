from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, ignore_index: int = 255) -> torch.Tensor:
    """
    Soft (multi-class) Dice loss. Operates on softmax probabilities.
    Args:
      logits — (B, C, H, W) raw network output.
      targets — (B, H, W) long class IDs in [0, C-1] (or ignore_index for unlabeled pixels).
      smooth — Laplace smoothing to avoid div-by-zero when a class is absent.
      ignore_index — pixels with this label are excluded from numerator and denominator.
    Returns: scalar Dice loss = 1 - mean over classes of 2·|P∩G| / (|P|+|G|).
    Notes:
      - Convert targets to one-hot (B, C, H, W) via F.one_hot then permute.
      - Mask out ignore_index pixels in BOTH probs and one-hot before reducing.
      - Sum over (B, H, W) per class, then average over classes.
    """
    valid = (targets != ignore_index)
    num_classes = logits.shape[1]
    indices = targets.masked_fill(~valid, 0)
    one_hot_tensor = F.one_hot(indices, num_classes=num_classes)
    one_hot_tensor = one_hot_tensor.permute(0, 3, 1, 2).float()
    probs = logits.softmax(dim=1)
    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    one_hot_tensor = one_hot_tensor * mask
    intersection = (probs * one_hot_tensor).sum(dim=(0, 2, 3))
    cardinality = probs.sum(dim=(0, 2, 3)) + one_hot_tensor.sum(dim=(0, 2, 3))
    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_per_class.mean()


class SegmentationLoss(nn.Module):
    """
    Sum of cross-entropy and Dice. CE pushes pixels toward correct class; Dice prevents
    degenerate solutions when classes are imbalanced (e.g. mostly background).
    """

    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = 255, class_weights: torch.Tensor | None = None) -> None:
        """
        Args:
          num_classes — total seg classes.
          ce_weight / dice_weight — scalar weights on each component.
          ignore_index — pixel label to skip in both losses (255 by convention).
          class_weights — optional (C,) tensor passed to F.cross_entropy for class re-balancing.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
          logits — (B, C, H, W) raw output.
          targets — (B, H, W) long class IDs.
        Returns: (total_loss, log_dict) where log_dict has 'loss', 'ce_loss', 'dice_loss'.
        Pipeline:
          1. ce = F.cross_entropy(logits, targets, weight=class_weights or None, ignore_index=self.ignore_index).
          2. dice = dice_loss(logits, targets, ignore_index=self.ignore_index).
          3. total = ce_weight * ce + dice_weight * dice.
        """
        weight = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=self.ignore_index)
        dice = dice_loss(logits, targets, ignore_index=self.ignore_index)
        total = self.ce_weight * ce + self.dice_weight * dice
        log = {"loss": total.item(), "ce_loss": ce.item(), "dice_loss": dice.item()}
        return total, log
