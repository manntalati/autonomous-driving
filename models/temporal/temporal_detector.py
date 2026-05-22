"""
Temporal-fusion detector (Phase 6).

A shared backbone runs on every frame of a window; a TemporalCrossAttention
module fuses the past frames' deepest features (C5) into the current frame's
C5; the Phase 2 FPN + detection head then run on the fused features. The FPN,
head, anchor generator and postprocess are reused unchanged from Phase 2 —
only the C5 feeding the FPN is temporally enriched.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from models.detection.detector import FPNDetector
from models.temporal.temporal_attention import TemporalCrossAttention


class TemporalDetector(nn.Module):
    """Wraps a Phase 2 FPNDetector with temporal cross-attention on C5."""

    def __init__(self, detector: FPNDetector, temporal_attn: TemporalCrossAttention) -> None:
        """
        Args:
          detector — a fully-built Phase 2 FPNDetector (backbone + FPN + head + anchors).
          temporal_attn — TemporalCrossAttention sized to the backbone's C5 channels.
        """
        super().__init__()
        self.detector = detector
        self.temporal_attn = temporal_attn

    def forward(self, frames: torch.Tensor, return_raw: bool = False):
        """
        Args:
          frames — (B, T, 3, H, W); frames[:, -1] is the current frame,
                   frames[:, :-1] the past frames (oldest→newest).
          return_raw — return raw outputs even in eval (val loss without BN drift).
        Returns:
          training / return_raw → (cls_logits, bbox_deltas, anchors);
          eval → postprocessed (boxes, scores, labels) per image.
        """
        T = frames.shape[1]
        c3s, c4s, c5s = [], [], []
        for t in range(T):
            c3, c4, c5 = self.detector.backbone(frames[:, t])
            c3s.append(c3)
            c4s.append(c4)
            c5s.append(c5)

        fused_c5 = self.temporal_attn(c5s[-1], c5s[:-1])

        fpn_features = self.detector.fpn((c3s[-1], c4s[-1], fused_c5))
        cls_logits, bbox_deltas = self.detector.head(list(fpn_features))

        feature_map_sizes = [(f.shape[-2], f.shape[-1]) for f in fpn_features]
        image_size = (frames.shape[-2], frames.shape[-1])
        anchors = self.detector.anchor_generator.generate_all(
            feature_map_sizes, image_size, device=frames.device
        )

        if self.training or return_raw:
            return cls_logits, bbox_deltas, anchors
        return self.detector.postprocess(cls_logits, bbox_deltas, anchors, image_size)
