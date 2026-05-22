"""
BEV detector (Phase 5): CAM_FRONT image → Lift-Splat-Shoot → BEV detection.

  image → ResNet backbone → C4 features
        → LiftSplatShoot   → top-down BEV feature grid
        → BEVEncoder       → refined BEV features
        → BEVDetectionHead → per-class centre heatmap + box regression
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

from models.backbone.resnet import ConvBlock, ResNetBackbone
from models.bev.lss import LiftSplatShoot

# The detector lifts the backbone's C4 feature map (stride 16, 256 channels).
_FEAT_STRIDE = 16
_FEAT_CHANNELS = 256


class BEVEncoder(nn.Module):
    """Small CNN that refines the splatted BEV grid before the detection head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVDetectionHead(nn.Module):
    """
    Centre-based BEV detection head. From the BEV feature grid it predicts:
      - heatmap    (B, num_classes, X, Y) — per-class object-centre probability.
      - regression (B, 6, X, Y) — sub-cell offset (2), size length/width (2),
        heading as (sin, cos) (2), read off at object-centre cells.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tower = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels),
        )
        self.heatmap = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.regression = nn.Conv2d(in_channels, 6, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x — (B, in_channels, X, Y) BEV features.
        Returns: (heatmap (B,num_classes,X,Y) in [0,1], regression (B,6,X,Y)).
        """
        feat = self.tower(x)
        return torch.sigmoid(self.heatmap(feat)), self.regression(feat)


class BEVDetector(nn.Module):
    """Full Phase 5 model: backbone + LSS + BEV encoder + detection head."""

    def __init__(self, num_classes: int = 3, image_size: Tuple[int, int] = (448, 800), xbound: Tuple[float, float, float] = (0.0, 51.2, 0.8), ybound: Tuple[float, float, float] = (-25.6, 25.6, 0.8), zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.0), dbound: Tuple[float, float, float] = (4.0, 50.0, 1.0), bev_channels: int = 64) -> None:
        super().__init__()
        self.backbone = ResNetBackbone()  # returns (C3, C4, C5)
        self.lss = LiftSplatShoot(
            in_channels=_FEAT_CHANNELS,
            image_size=image_size,
            feat_stride=_FEAT_STRIDE,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            bev_channels=bev_channels,
        )
        self.encoder = BEVEncoder(bev_channels)
        self.head = BEVDetectionHead(bev_channels, num_classes)

    def load_pretrained(self) -> None:
        """Load ImageNet weights into the ResNet backbone."""
        self.backbone.load_pretrained()

    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor,
                cam_to_ego: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          images — (B, 3, H, W).
          intrinsics — (B, 3, 3) camera K.
          cam_to_ego — (B, 4, 4) camera→ego transform.
        Returns: (heatmap (B,num_classes,X,Y), regression (B,6,X,Y)).
        """
        _, c4, _ = self.backbone(images)
        bev = self.lss(c4, intrinsics, cam_to_ego)
        bev = self.encoder(bev)
        return self.head(bev)
