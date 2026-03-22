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
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.tower = nn.Sequential(*[
            layer for _ in range(num_convs)
            for layer in (nn.Conv2d(in_channels, in_channels, 3, padding=1), 
                          nn.ReLU())
        ])

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