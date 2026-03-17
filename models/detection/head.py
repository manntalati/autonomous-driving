import torch.nn as nn
import torch
from typing import Tuple, List

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = 4) -> None:
        """
        Build a shared conv tower then split into:
        - cls_head: predicts (num_anchors * num_classes) per location
        - reg_head: predicts (num_anchors * 4) per location
        """
        pass

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply head to each FPN level independently.
        Returns:
          cls_logits: list of (B, H*W*A, num_classes) per level
          bbox_deltas: list of (B, H*W*A, 4) per level
        """
        pass
