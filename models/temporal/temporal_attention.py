"""
Temporal cross-attention (Phase 6).

The current frame's feature map is the QUERY; the past frames' feature maps
form a KEY/VALUE memory. Cross-attention lets each current-frame location
pull in evidence from where things were in earlier frames — so a car that is
briefly occluded or blurry now can still be supported by its past appearance.

A learned per-frame temporal embedding tags each past frame so the module can
distinguish t-1 from t-2.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


class TemporalCrossAttention(nn.Module):
    """
    Multi-head cross-attention from the current feature map to a memory built
    from the past feature maps. Output is residual: current + attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_past_frames: int = 2) -> None:
        """
        Args:
          embed_dim — channel count of the feature map being fused (C5 → 512).
          num_heads — attention heads (embed_dim must be divisible by num_heads).
          num_past_frames — how many past frames form the memory.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # one learned embedding per past frame, added to that frame's K/V tokens
        self.temporal_embed = nn.Parameter(torch.zeros(num_past_frames, embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, current: torch.Tensor, past: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
          current — (B, C, H, W) current-frame feature map.
          past — list of (B, C, H, W) past-frame feature maps, ordered oldest→newest;
                 len(past) must equal num_past_frames.
        Returns: (B, C, H, W) temporally-fused current features.
        """
        B, C, H, W = current.shape
        N = H * W
        q_tokens = current.flatten(2).transpose(1, 2)

        memory_tokens = []
        for i, frame in enumerate(past):
            tok = frame.flatten(2).transpose(1, 2)
            memory_tokens.append(tok + self.temporal_embed[i])
        memory = torch.cat(memory_tokens, dim=1)

        q = self.q_proj(q_tokens)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)

        def _heads(x):
            return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = _heads(q), _heads(k), _heads(v)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        fused = self.norm(q_tokens + out)
        return fused.transpose(1, 2).reshape(B, C, H, W)
