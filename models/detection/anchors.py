import torch
from typing import List, Tuple
import math

class AnchorGenerator:
    def __init__(self, scales: List[float], aspect_ratios: List[float], strides: List[int]) -> None:
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    def generate_for_level(self, feature_h: int, feature_w: int, stride: int, scale: float) -> torch.Tensor:
        """
        Tile anchors across a single feature map level.
        Returns: (H*W*num_anchors, 4) in [x1,y1,x2,y2]
        """
        tile_anchors = []
        for ratio in self.aspect_ratios:
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            for row in range(feature_h):
                for col in range(feature_w):
                    cx = (col + 0.5) * stride
                    cy = (row + 0.5) * stride
                    x1, x2 = cx - w/2, cx + w/2
                    y1, y2 = cy - h/2, cy + h/2
                    tile_anchors.append([x1,y1,x2,y2])
        return torch.tensor(tile_anchors)

    def generate_all(self, feature_map_sizes: List[Tuple[int, int]], image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate anchors for all FPN levels and concatenate.
        Returns: (total_anchors, 4)
        """
        anchors = []
        for i, (h, w) in enumerate(feature_map_sizes):
            tile_anchors = self.generate_for_level(h, w, self.strides[i], self.scales[i])
            anchors.append(tile_anchors)
        anchors = torch.cat(anchors, dim=0)
        img_h, img_w = image_size
        anchors[:, 0].clamp_(min=0, max=img_w)
        anchors[:, 1].clamp_(min=0, max=img_h)
        anchors[:, 2].clamp_(min=0, max=img_w)
        anchors[:, 3].clamp_(min=0, max=img_h)
        return anchors
