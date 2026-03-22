import torch
from typing import Tuple, List
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        """
        Build lateral 1x1 convs (to unify channels) and
        top-down 3x3 convs (after upsampling + adding).
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(channel, out_channels, kernel_size=1) for channel in in_channels])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels])

    def forward(self, features: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Takes (C3, C4, C5), returns (P3, P4, P5).
        Top-down: upsample P5 → add to C4 → P4, upsample P4 → add to C3 → P3.
        """
        c3, c4, c5 = features
        lateral5, lateral4, lateral3 = self.lateral_convs[2](c5), self.lateral_convs[1](c4), self.lateral_convs[0](c3)
        p5 = lateral5 
        p4 = lateral4 + F.interpolate(p5, size=lateral4.shape[-2:], mode='nearest') 
        p3 = lateral3 + F.interpolate(p4, size=lateral3.shape[-2:], mode='nearest')
        p5_ret, p4_ret, p3_ret = self.output_convs[2](p5), self.output_convs[1](p4), self.output_convs[0](p3)
        return (p5_ret, p4_ret, p3_ret)