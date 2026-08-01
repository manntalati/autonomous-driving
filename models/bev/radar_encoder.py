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
per-cell operation — no resampling, no calibration work. That is the payoff of
the Phase 5 Lift-Splat-Shoot transform: it turns radar fusion from a research
project into a ten-line module.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RadarBEVEncoder(nn.Module):
    """
    Encode the sparse rasterized radar BEV grid into a dense feature map.

    Args:
        in_channels: rasterized radar channels (5, see RADAR_BEV_CHANNELS).
        out_channels: feature channels to emit; match the camera BEV width so
            fusion can gate without a projection.
        hidden: width of the intermediate conv.

    Shapes: (B, in_channels, X, Y) -> (B, out_channels, X, Y). Resolution is
    preserved — the shared BEVEncoder downsamples afterwards.

    Deliberately shallow (3 convs). The input is >98% zeros and there are only
    ~3,300 training frames; a deep encoder memorises clutter. Input scaling is
    handled by fixed constants in `rasterize_radar_bev` rather than BatchNorm,
    because batch statistics over a mostly-empty tensor are dominated by empty
    cells and drift between day and night frames.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, radar_bev: torch.Tensor) -> torch.Tensor:
        """Args: (B, in_channels, X, Y). Returns: (B, out_channels, X, Y)."""
        if radar_bev.dim() != 4:
            raise ValueError(f"expected (B, C, X, Y), got {tuple(radar_bev.shape)}")
        return self.encoder(radar_bev)


class CameraRadarFusion(nn.Module):
    """
    Fuse the camera BEV feature grid with the radar BEV feature grid.

    Args:
        channels: feature width of both inputs (must match).
        mode: "concat" | "gated".

    Shapes: (B, C, X, Y) x (B, C, X, Y) -> (B, C, X, Y).

    MODES
    -----
    "concat": channel-concatenate then 1x1 conv back to `channels`. Simple, strong
        baseline, the thing to beat.

    "gated": learn a per-cell, per-channel gate from both streams,
            g = sigmoid(conv([cam, radar]))
            out = g * cam + (1 - g) * radar
        A fixed-weight sum forces equal trust in camera and radar everywhere.
        A gate lets the model learn to *down-weight the camera where the camera is
        unreliable* — precisely the day/night behaviour under test. It also gives
        a free interpretability probe: if the thesis holds, the mean gate should
        shift measurably toward radar on night frames.

    WHAT WOULD FALSIFY THE THESIS
    -----------------------------
    If night mAP improves but the learned gate does NOT shift toward radar at
    night, the gain is added capacity rather than radar's illumination
    invariance. Report that if it happens — Phase 4 already set the precedent
    with the ViT-vs-CNN tie.
    """

    def __init__(self, channels: int = 64, mode: str = "gated") -> None:
        super().__init__()
        if mode not in ("concat", "gated"):
            raise ValueError(f"mode must be 'concat' or 'gated', got {mode!r}")
        self.channels = channels
        self.mode = mode
        self._last_gate: Optional[torch.Tensor] = None

        if mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.gate = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, cam_bev: torch.Tensor, radar_bev: torch.Tensor) -> torch.Tensor:
        """Args: two (B, C, X, Y) grids. Returns: (B, C, X, Y) fused grid."""
        if cam_bev.shape != radar_bev.shape:
            raise ValueError(
                f"camera and radar grids must match, got {tuple(cam_bev.shape)} "
                f"and {tuple(radar_bev.shape)}"
            )
        stacked = torch.cat([cam_bev, radar_bev], dim=1)
        if self.mode == "concat":
            self._last_gate = None
            return self.fuse(stacked)
        g = self.gate(stacked)
        # detach: this is instrumentation, not a gradient path. Keeping it
        # attached would hold the graph alive and leak memory across an epoch.
        self._last_gate = g.detach()
        return g * cam_bev + (1.0 - g) * radar_bev

    def last_gate(self) -> Optional[torch.Tensor]:
        """Most recent gate tensor (B, C, X, Y) detached, or None in concat mode."""
        return self._last_gate

    def mean_gate(self) -> Optional[float]:
        """
        Scalar mean of the last gate — the day/night gate-shift statistic.
        Values near 1 mean the model is relying on the camera; near 0, on radar.
        """
        if self._last_gate is None:
            return None
        return float(self._last_gate.mean())
