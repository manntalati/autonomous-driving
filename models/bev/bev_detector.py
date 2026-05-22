"""
BEV detector (Phase 5): CAM_FRONT image → Lift-Splat-Shoot → BEV detection.

  image → ResNet backbone → C4 features
        → LiftSplatShoot   → top-down BEV feature grid
        → BEVEncoder       → refined BEV features
        → BEVDetectionHead → per-class centre heatmap + box regression
"""
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BEVSegHead(nn.Module):
    """
    BEV semantic-map head (Phase 8). From the BEV feature grid it predicts a
    per-cell class map — background / drivable / lane / ped_crossing / walkway.
    """

    def __init__(self, in_channels: int, num_seg_classes: int) -> None:
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.net = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, num_seg_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, in_channels, X, Y). Returns: (B, num_seg_classes, X, Y) logits."""
        return self.net(x)


class BEVDetector(nn.Module):
    """Phase 5/8 model: backbone + LSS + BEV encoder + detection head + seg head."""

    def __init__(self, num_classes: int = 3, image_size: Tuple[int, int] = (448, 800), xbound: Tuple[float, float, float] = (0.0, 51.2, 0.8), ybound: Tuple[float, float, float] = (-25.6, 25.6, 0.8), zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.0), dbound: Tuple[float, float, float] = (4.0, 50.0, 1.0), bev_channels: int = 64, num_seg_classes: int = 5) -> None:
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
        self.seg_head = BEVSegHead(bev_channels, num_seg_classes)

    def load_pretrained(self) -> None:
        """Load ImageNet weights into the ResNet backbone."""
        self.backbone.load_pretrained()

    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor,
                cam_to_ego: torch.Tensor) -> dict:
        """
        Args:
          images — (B, N, 3, H, W) — N surround cameras (N=1 for single-camera).
          intrinsics — (B, N, 3, 3) camera K.
          cam_to_ego — (B, N, 4, 4) camera→ego transform.
        Returns: dict with
          'heatmap'    (B, num_classes, X, Y)     — object-centre probabilities,
          'regression' (B, 6, X, Y)               — box regression,
          'seg'        (B, num_seg_classes, X, Y) — BEV semantic-map logits,
          'depth'      (B, D, Hf, Wf)             — reference-camera depth distribution.
        """
        B, N = images.shape[:2]
        _, c4, _ = self.backbone(images.flatten(0, 1))   # (B·N, 256, Hf, Wf)
        c4 = c4.reshape(B, N, *c4.shape[1:])             # (B, N, 256, Hf, Wf)
        bev_grid, depth = self.lss(c4, intrinsics, cam_to_ego, return_depth=True)
        bev = self.encoder(bev_grid)
        heatmap, regression = self.head(bev)
        return {
            "heatmap": heatmap, "regression": regression,
            "seg": self.seg_head(bev),
            "depth": depth[:, 0],   # reference camera's depth distribution
        }


def decode_bev_detections(
    heatmap: torch.Tensor,
    regression: torch.Tensor,
    xbound,
    ybound,
    score_threshold: float = 0.3,
    max_detections: int = 50,
):
    """
    Decode the BEV head outputs into BEV boxes — the inverse of encode_bev_targets,
    used for the Phase 7 / P5-4 top-down visualization.
    Args:
      heatmap — (num_classes, X, Y) per-class centre probabilities (already sigmoid).
      regression — (6, X, Y) [offset_x, offset_y, length, width, sin yaw, cos yaw].
      xbound/ybound — BEV grid [lower, upper, cell_size] per axis.
      score_threshold — discard peaks below this.
      max_detections — keep at most this many, by score.
    Returns: (boxes (N,5)=[x,y,length,width,yaw], scores (N,), labels (N,)).
    Pipeline:
      1. find peak cells — local maxima of the heatmap (e.g. 3×3 max-pool == heatmap)
         above score_threshold.
      2. at each peak (class c, cell ix,iy): read regression →
         x = (ix + offset_x)·x_cell + x_lower ; y likewise ; yaw = atan2(sin, cos).
      3. sort by score, keep the top max_detections.
    """
    # peaks = cells equal to their 3×3 local max and above the score threshold
    pooled = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    peaks = (heatmap == pooled) & (heatmap >= score_threshold)
    idx = peaks.nonzero(as_tuple=False)  # (P, 3) — (class, ix, iy)

    if idx.shape[0] == 0:
        return (torch.zeros(0, 5), torch.zeros(0), torch.zeros(0, dtype=torch.long))

    x_lo, _, x_cell = xbound
    y_lo, _, y_cell = ybound
    boxes, scores, labels = [], [], []
    for c, ix, iy in idx.tolist():
        offx, offy, length, width, sin, cos = regression[:, ix, iy].tolist()
        x = (ix + offx) * x_cell + x_lo
        y = (iy + offy) * y_cell + y_lo
        boxes.append([x, y, length, width, math.atan2(sin, cos)])
        scores.append(float(heatmap[c, ix, iy]))
        labels.append(int(c))

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    if scores.numel() > max_detections:
        topk = torch.topk(scores, max_detections)
        boxes, scores, labels = boxes[topk.indices], topk.values, labels[topk.indices]
    return boxes, scores, labels
