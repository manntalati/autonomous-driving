import torch
from typing import Tuple
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        """
        Build lateral 1x1 convs (to unify channels) and
        top-down 3x3 convs (after upsampling + adding).
        """
        pass

    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Takes (C3, C4, C5), returns (P3, P4, P5).
        Top-down: upsample P5 → add to C4 → P4, upsample P4 → add to C3 → P3.
        """
        pass
