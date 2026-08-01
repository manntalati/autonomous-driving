from collections import OrderedDict
import torch.nn as nn
import torch
from typing import Tuple, List

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = 4,
                 dropout: float = 0.0) -> None:
        """
        Build a shared conv tower then split into:
        - cls_head: predicts (num_anchors * num_classes) per location
        - reg_head: predicts (num_anchors * 4) per location

        Args:
            dropout: Dropout2d probability applied after each ReLU in the tower.
                Default 0.0 (no-op) preserves the original behaviour exactly.
                Set > 0 and fine-tune to enable MC-dropout uncertainty (Phase 11).

        CHECKPOINT COMPATIBILITY
        ------------------------
        The tower is built with explicit OrderedDict names so the conv layers keep
        the keys they had before dropout existed ("tower.0.*", "tower.2.*", ...).
        Naively inserting Dropout2d into a positional nn.Sequential would renumber
        every layer after it and silently break every checkpoint on disk. Dropout
        modules hold no parameters, so they add nothing to the state dict and the
        keys are byte-identical whether or not dropout is enabled.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

        layers: list[tuple[str, nn.Module]] = []
        for i in range(num_convs):
            # names "0","1","2","3",... reproduce the original positional indices
            layers.append((str(2 * i), nn.Conv2d(in_channels, in_channels, 3, padding=1)))
            layers.append((str(2 * i + 1), nn.ReLU()))
            # parameter-free, so it never appears in the state dict
            layers.append((f"drop{i}", nn.Dropout2d(dropout)))
        self.tower = nn.Sequential(OrderedDict(layers))

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply head to each FPN level independently.
        Returns:
          cls_logits: list of (B, H*W*A, num_classes) per level
          bbox_deltas: list of (B, H*W*A, 4) per level
        """
        cls_output, reg_output = [], []
        for feature in features:
          x = self.tower(feature)
          B, _, H, W = feature.shape
          cls_logits = self.cls_head(x).permute(0,2,3,1).reshape(B, H*W*self.num_anchors, self.num_classes)
          bbox_deltas = self.reg_head(x).permute(0,2,3,1).reshape(B, H*W*self.num_anchors, 4)
          cls_output.append(cls_logits)
          reg_output.append(bbox_deltas)
        return (cls_output, reg_output)
