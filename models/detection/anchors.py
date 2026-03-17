import torch
from typing import List, Tuple

class AnchorGenerator:
    def __init__(self, scales: List[float], aspect_ratios: List[float], strides: List[int]) -> None:
        pass

    def generate_for_level(self, feature_h: int, feature_w: int, stride: int, scale: float) -> torch.Tensor:
        """
        Tile anchors across a single feature map level.
        Returns: (H*W*num_anchors, 4) in [x1,y1,x2,y2]
        """
        pass

    def generate_all(self, feature_map_sizes: List[Tuple[int, int]], image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate anchors for all FPN levels and concatenate.
        Returns: (total_anchors, 4)
        """
        pass
