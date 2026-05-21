from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import ConvBlock, ResNetBackbone


class UpBlock(nn.Module):
    """
    One decoder step: bilinear 2× upsample → concat with skip → two ConvBlocks.
    Used to walk back up the resolution pyramid C5 → C4 → C3 → input.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Args:
          in_channels — channel count of the upsampled feature coming from the deeper decoder stage.
          skip_channels — channel count of the encoder skip connection at the matching resolution.
          out_channels — channel count produced by this block.
        Notes:
          - First conv consumes (in_channels + skip_channels) after concat.
          - No transposed conv — bilinear upsample is parameter-free and avoids checkerboard artifacts.
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
          x — (B, in_channels, H, W) deeper feature.
          skip — (B, skip_channels, 2H, 2W) encoder skip at higher resolution,
                 or None for a pure 2× upsample (skip_channels must be 0).
        Returns: (B, out_channels, 2H, 2W).
        Notes:
          - With a skip: upsample x to skip's spatial size, concat along channels.
          - Without a skip: upsample x by 2×. Then conv1 → conv2.
        """
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(x))


class UNet(nn.Module):
    """
    U-Net segmentation head on top of the Phase 1 ResNet backbone.
    Encoder = ResNetBackbone (C3=128/stride 8, C4=256/stride 16, C5=512/stride 32).
    Decoder upsamples back to input resolution and produces (B, num_classes, H, W) logits.
    """

    def __init__(self, backbone: ResNetBackbone, num_classes: int = 5, decoder_channels: Tuple[int, int, int] = (256, 128, 64)) -> None:
        """
        Args:
          backbone — ResNetBackbone returning (C3, C4, C5). Must NOT have a classifier head (num_classes=None).
          num_classes — number of segmentation classes (incl. background).
          decoder_channels — (d3, d2, d1) output channels of each UpBlock, coarse→fine.
        Notes:
          - d3: applied at C4 resolution after merging C5 ↑ with C4 skip.
          - d2: applied at C3 resolution after merging d3 ↑ with C3 skip.
          - d1: applied at stride-4 resolution (no skip — pure upsample of d2).
          - Final 1×1 conv projects to num_classes, then bilinear upsample 4× to input resolution.
        """
        super().__init__()
        assert backbone.num_classes is None, "U-Net expects feature-map backbone, not a classifier."
        self.backbone = backbone
        d3, d2, d1 = decoder_channels

        self.up3 = UpBlock(in_channels=512, skip_channels=256, out_channels=d3)  # C5 + C4 -> stride 16
        self.up2 = UpBlock(in_channels=d3,  skip_channels=128, out_channels=d2)  # ^ + C3 -> stride 8
        self.up1 = UpBlock(in_channels=d2,  skip_channels=0,   out_channels=d1)  # pure upsample -> stride 4

        self.classifier = nn.Conv2d(d1, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x — (B, 3, H, W) image.
        Returns: (B, num_classes, H, W) per-pixel logits at input resolution.
        Pipeline:
          1. (c3, c4, c5) = self.backbone(x).
          2. d3 = self.up3(c5, c4).
          3. d2 = self.up2(d3, c3).
          4. d1 = self.up1(d2, skip=zeros_like at 2× resolution)   # or: handle skip=None in UpBlock.
          5. logits = self.classifier(d1).
          6. F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False) — final 4× upsample.
        """
        c3, c4, c5 = self.backbone(x)
        d3 = self.up3(c5, c4)
        d2 = self.up2(d3, c3)
        d1 = self.up1(d2)
        logits = self.classifier(d1)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
