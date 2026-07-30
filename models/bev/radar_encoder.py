"""
P10-3 — Radar BEV encoder and camera-radar fusion.

Architecture:

    images (B,N,3,H,W) ──> ResNet ──> LiftSplatShoot ──> cam_bev  (B, C_bev, X, Y)
                                                              │
    radar_bev (B,5,X,Y) ──> RadarBEVEncoder ──> rad_bev (B, C_r, X, Y)
                                                              │
                                        CameraRadarFusion ────┴──> fused (B, C_bev, X, Y)
                                                              │
                                              BEVEncoder ──> BEVDetectionHead

Both branches already live in the same ego-frame BEV grid, so fusion is a plain
per-cell operation — no resampling, no calibration work. This is the main reason
to fuse in BEV rather than in image space, and it is worth stating explicitly in
the write-up: the Lift-Splat-Shoot BEV grid from Phase 5 is what makes radar
fusion a ten-line module instead of a research project.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RadarBEVEncoder(nn.Module):
    """
    Encode the sparse rasterized radar BEV grid into a dense feature map.

    Args:
        in_channels: number of rasterized radar channels (5, see RADAR_BEV_CHANNELS).
        out_channels: feature channels to emit; match the camera BEV width so the
            fusion module can add or gate without a projection.

    Shapes: (B, in_channels, X, Y) -> (B, out_channels, X, Y). The grid resolution
    is preserved — do not downsample here; the fusion happens at full BEV
    resolution and the shared BEVEncoder downsamples afterwards.

    IMPLEMENTATION NOTES
    --------------------
    Keep this small — 2-3 conv blocks. The radar input is >98% zeros and there are
    only ~3,300 training frames; a deep encoder will memorise clutter.

    Normalise the input channels before the first conv. They have wildly different
    scales: occupancy is a small count (0-10), rcs is in dBsm (roughly -25..+40),
    and velocities are m/s (-20..+20). Feeding those to a conv unnormalised lets
    rcs dominate the gradient purely because of its magnitude. Either use a
    BatchNorm2d on the input or divide by fixed per-channel constants — prefer the
    fixed constants so inference does not depend on batch statistics from a
    98%-zero tensor, where BatchNorm's running estimates are dominated by empty
    cells and drift badly between day and night frames.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 64) -> None:
        super().__init__()
        raise NotImplementedError("P10-3")

    def forward(self, radar_bev: torch.Tensor) -> torch.Tensor:
        """Args: (B, in_channels, X, Y). Returns: (B, out_channels, X, Y)."""
        raise NotImplementedError("P10-3")


class CameraRadarFusion(nn.Module):
    """
    Fuse the camera BEV feature grid with the radar BEV feature grid.

    Args:
        channels: feature width of both inputs (they must match).
        mode: "concat" | "gated" — see below.

    Shapes: (B, C, X, Y) x (B, C, X, Y) -> (B, C, X, Y).

    FUSION MODES (implement "concat" first; "gated" is the interesting one)
    ----------------------------------------------------------------------
    "concat": channel-concatenate then 1x1 conv back down to `channels`. Simple,
        strong baseline, and the thing to beat. Report it.

    "gated": learn a per-cell, per-channel gate from both streams,
            g = sigmoid(conv([cam, radar]))
            out = g * cam + (1 - g) * radar
        This matters for the Phase 10 thesis. A fixed-weight sum forces the model
        to trust camera and radar equally everywhere. A gate lets it learn to
        *down-weight the camera where the camera is unreliable* — which is
        precisely the day/night behaviour under test. It also gives Phase 11 a
        free interpretability probe: if the thesis is right, mean gate value
        should shift measurably toward radar on night frames. Log the mean gate
        per frame and plot it against the day/night label; that single figure is
        the most convincing artifact this phase can produce.

    A NOTE ON WHAT WOULD FALSIFY THE THESIS
    ---------------------------------------
    If night mAP improves and the learned gate does NOT shift toward radar at
    night, the gain is coming from added capacity rather than from radar's
    illumination invariance. Report that honestly if it happens — a well-measured
    negative result is worth far more than a hand-waved positive one, and Phase 4
    already set that precedent with the ViT-vs-CNN tie.
    """

    def __init__(self, channels: int = 64, mode: str = "gated") -> None:
        super().__init__()
        raise NotImplementedError("P10-3")

    def forward(self, cam_bev: torch.Tensor, radar_bev: torch.Tensor) -> torch.Tensor:
        """Args: two (B, C, X, Y) grids. Returns: (B, C, X, Y) fused grid."""
        raise NotImplementedError("P10-3")

    def last_gate(self) -> torch.Tensor | None:
        """
        The most recent gate tensor (B, C, X, Y), or None in "concat" mode.

        Cache it in forward() so training/eval code can log mean gate values
        without a second pass. Detach it — this is instrumentation, not a
        gradient path, and keeping it attached will hold the graph alive and
        leak memory across the epoch.
        """
        raise NotImplementedError("P10-3")
