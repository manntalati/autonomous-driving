import torch
import torch.nn as nn
from models.backbone.resnet import ResNetBackbone
from models.detection.fpn import FPN
from models.detection.head import DetectionHead
from models.detection.anchors import AnchorGenerator
from typing import Tuple, List

class FPNDetector(nn.Module):
    def __init__(self, backbone: ResNetBackbone, fpn: FPN, head: DetectionHead, anchor_generator: AnchorGenerator, num_classes: int, score_threshold: float = 0.05, nms_threshold: float = 0.5, max_detections: int = 100) -> None:
        pass

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Training mode: returns (cls_logits, bbox_deltas, anchors)
        Eval mode: calls postprocess() and returns (boxes, scores, labels) per image
        """
        pass

    def postprocess(self, cls_logits: List[torch.Tensor], bbox_deltas: List[torch.Tensor], anchors: torch.Tensor, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Decode deltas → absolute boxes
        2. Sigmoid scores, threshold
        3. Apply NMS per class
        4. Return top-k detections
        """
        pass
